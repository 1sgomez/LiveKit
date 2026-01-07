# voice_agent.py - Agente de voz 100% local con LiveKit
# Uso:
#   Modo worker: python voice_agent.py (espera trabajos de LiveKit)
#   Modo directo: python voice_agent.py --direct (se conecta directamente a la sala)

import asyncio
import logging
import warnings
import os
import sys
import tempfile
import argparse
from typing import Optional
import numpy as np
import requests

# Importaciones de LiveKit
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero

# Importaciones de modelos
import whisper
from gtts import gTTS

from dotenv import load_dotenv

load_dotenv()

# === CONFIGURACIÓN ===
os.environ['LIVEKIT_LOG_LEVEL'] = 'error'
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logging.getLogger('livekit.agents').setLevel(logging.ERROR)

# Configuración de LiveKit
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
LIVEKIT_ROOM = os.getenv('LIVEKIT_ROOM', 'test')


class LocalVoiceAgent:
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.ctx_room = None
        self.conversation_history = []
        self.audio_buffer = []
        self.is_processing = False
        self.silence_detected = False
        self.greeting_sent = False

        logger.info("📥 Cargando Whisper (STT)...")
        self.whisper_model = whisper.load_model("base")

        logger.info("📥 Cargando VAD (Silero)...")
        self.vad = silero.VAD.load()

        logger.info("✅ Modelos cargados correctamente")

    async def connect(self, url: str, token: str):
        """Conectar directamente a una sala"""
        logger.info(f"🔌 Conectando a la sala {self.room_name}...")

        # Crear Room de LiveKit
        self.ctx_room = rtc.Room()

        # Configurar eventos
        self.ctx_room.on("track_subscribed", self.on_track_subscribed)
        self.ctx_room.on("participant_connected", self.on_participant_connected)
        self.ctx_room.on("connected", self.on_connected)

        # Conectar
        await self.ctx_room.connect(url, token)

        logger.info(f"✅ Conectado a {self.room_name}")
        logger.info(f"👤 Identity: {self.ctx_room.local_participant.identity}")

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribir audio con Whisper"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.whisper_model.transcribe(audio_np, language="es")
            text = result["text"].strip()

            if text:
                logger.info(f"👤 Usuario: {text}")
            return text

        except Exception as e:
            logger.error(f"❌ Error en transcripción: {e}")
            return ""

    async def generate_response(self, user_message: str) -> str:
        """Generar respuesta con Ollama"""
        try:
            self.conversation_history.append({"role": "user", "content": user_message})

            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2:3b",
                    "messages": [
                        {"role": "system", "content": "Eres Jarvis, asistente de voz conciso en español."},
                        *self.conversation_history
                    ],
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result["message"]["content"].strip()
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                logger.info(f"🤖 Jarvis: {assistant_message}")
                return assistant_message

            return "Disculpa, tuve un problema."

        except Exception as e:
            logger.error(f"❌ Error con Ollama: {e}")
            return "No puedo conectar con el servidor de IA."

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Sintetizar voz con gTTS"""
        try:
            logger.info(f"🔊 Generando audio: {text[:50]}...")

            tts = gTTS(text=text, lang='es', slow=False)

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                temp_file = fp.name
                tts.save(temp_file)

                with open(temp_file, 'rb') as f:
                    audio_data = f.read()

                os.unlink(temp_file)

            return audio_data

        except Exception as e:
            logger.error(f"❌ Error sintetizando: {e}")
            return None

    async def publish_audio(self, audio_data: bytes):
        """Publicar audio en la sala"""
        try:
            sample_rate = 22050
            source = rtc.AudioSource(sample_rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)

            options = rtc.TrackPublishOptions()
            await self.ctx_room.local_participant.publish_track(track, options)
            logger.info("✅ Audio publicado")

            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            chunk_size = 2048
            for i in range(0, len(audio_np), chunk_size):
                chunk = audio_np[i:i+chunk_size]
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk)
                )
                await source.capture_frame(frame)
                await asyncio.sleep(len(chunk) / sample_rate)

            logger.info("✅ Audio reproducido")

        except Exception as e:
            logger.error(f"❌ Error publicando: {e}")

    async def send_greeting(self):
        """Enviar saludo"""
        if self.greeting_sent:
            return

        greeting = "¡Hola! Soy Jarvis, tu asistente de voz. ¿En qué puedo ayudarte?"
        logger.info("🎤 Enviando saludo...")

        audio = await self.synthesize_speech(greeting)
        if audio:
            await self.publish_audio(audio)
            self.greeting_sent = True

    async def handle_audio_stream(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """Procesar audio del usuario"""
        if self.is_processing:
            return

        self.is_processing = True
        self.audio_buffer = []
        self.silence_detected = False
        silence_start = None

        logger.info(f"🎤 Procesando de {participant.identity}")

        audio_stream = rtc.AudioStream(track)

        try:
            async for frame_event in audio_stream:
                current_time = asyncio.get_event_loop().time()

                audio_data = np.frombuffer(
                    frame_event.frame.data,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0

                is_speech = self.vad.is_speech(audio_data, sample_rate=16000)

                if is_speech:
                    self.audio_buffer.append(frame_event.frame.data)
                    self.silence_detected = False
                    silence_start = None
                else:
                    if not self.silence_detected:
                        self.silence_detected = True
                        silence_start = current_time
                    elif silence_start and (current_time - silence_start) >= 1.0 and len(self.audio_buffer) > 0:
                        logger.info("🛑 Silencio detectado")
                        break

            if len(self.audio_buffer) > 0:
                audio_bytes = b''.join(self.audio_buffer)

                text = await self.transcribe_audio(audio_bytes)
                if text:
                    response = await self.generate_response(text)
                    audio_response = await self.synthesize_speech(response)
                    if audio_response:
                        await self.publish_audio(audio_response)

        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.is_processing = False
            self.audio_buffer = []

    def on_track_subscribed(self, track: rtc.Track, publication, participant: rtc.RemoteParticipant):
        """Callback: track recibido"""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"🎧 Audio de {participant.identity}")
            asyncio.ensure_future(self.handle_audio_stream(track, participant))

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Callback: participante conectado"""
        logger.info(f"👋 {participant.identity} se unió")
        asyncio.ensure_future(self.send_greeting())

    def on_connected(self):
        """Callback: conectado"""
        logger.info("🔗 Conexión establecida")

        if self.ctx_room and len(self.ctx_room.remote_participants) > 0:
            asyncio.ensure_future(self.send_greeting())

    async def run(self):
        """Mantener agente activo"""
        logger.info("⏳ Escuchando... (Ctrl+C para salir)")
        await asyncio.sleep(float('inf'))


async def check_dependencies():
    """Verificar dependencias"""
    logger.info("🔍 Verificando dependencias...")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("✅ Ollama disponible")
        else:
            logger.warning("⚠️ Ollama con error")
    except Exception as e:
        logger.error(f"❌ Ollama no disponible: {e}")
        return False

    try:
        import whisper
        logger.info("✅ Whisper disponible")
    except ImportError:
        logger.error("❌ Whisper no instalado")
        return False

    return True


async def run_direct_mode():
    """Modo directo: conectar directamente a la sala"""
    logger.info("="*60)
    logger.info("🚀 MODO DIRECTO - Conectando directamente a la sala")
    logger.info("="*60)

    if not await check_dependencies():
        logger.error("❌ Dependencias no disponibles")
        return

    # Obtener token directamente
    import time
    participant_name = f"agent-{int(time.time())}"

    logger.info(f"📡 Solicitando token para sala '{LIVEKIT_ROOM}'...")

    try:
        response = requests.get(
            f"http://localhost:5000/token",
            params={"room": LIVEKIT_ROOM, "name": participant_name},
            timeout=10
        )

        if response.status_code != 200:
            logger.error(f"❌ Error obteniendo token: {response.status_code}")
            return

        data = response.json()
        token = data["token"]
        url = data["url"]

        logger.info(f"✅ Token recibido")
        logger.info(f"🔗 URL: {url}")

        # Crear agente y conectar
        agent = LocalVoiceAgent(LIVEKIT_ROOM)
        await agent.connect(url, token)
        await agent.run()

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def entrypoint(ctx: JobContext):
    """Entry point modo worker"""
    logger.info("="*60)
    logger.info("🚀 TRABAJO RECIBIDO - Modo worker")
    logger.info("="*60)

    if ctx is None:
        logger.error("❌ Error: ctx es None")
        return

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        if not ctx.room:
            logger.error("❌ No hay sala")
            return

        if not await check_dependencies():
            return

        agent = LocalVoiceAgent(ctx.room.name)
        agent.ctx_room = ctx.room

        # Configurar callbacks
        ctx.room.on("track_subscribed", agent.on_track_subscribed)
        ctx.room.on("participant_connected", agent.on_participant_connected)
        ctx.room.on("connected", agent.on_connected)

        logger.info(f"📍 Conectado a: {ctx.room.name}")
        logger.info("⏳ Esperando participantes...")
        await asyncio.sleep(float('inf'))

    except Exception as e:
        logger.error(f"❌ Error: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Agente de voz Jarvis")
    parser.add_argument("--direct", action="store_true",
                       help="Modo directo: conectar directamente a la sala")
    args = parser.parse_args()

    print('='*60)
    print('🚀 Voice Agent - Jarvis')
    print('='*60)

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"   LIVEKIT_URL: {LIVEKIT_URL}")
    print(f"   LIVEKIT_ROOM: {LIVEKIT_ROOM}")
    print()

    # Verificar configuración
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("❌ ERROR: Faltan LIVEKIT_API_KEY o LIVEKIT_API_SECRET")
        sys.exit(1)

    if args.direct:
        # Modo directo
        print("🎯 MODO: Directo (conectar directamente a la sala)")
        print("   El agente se conectará inmediatamente a la sala")
        print()

        try:
            asyncio.run(run_direct_mode())
        except KeyboardInterrupt:
            print("\n👋 Agente detenido")
    else:
        # Modo worker (espera trabajos)
        print("🎯 MODO: Worker (esperando trabajos de LiveKit)")
        print("   El agente esperará a que LiveKit le envíe trabajos")
        print("   NOTA: Tu versión de LiveKit puede no soportar workers")
        print()

        try:
            cli.run_app(
                WorkerOptions(
                    entrypoint_fnc=entrypoint,
                    api_key=LIVEKIT_API_KEY,
                    api_secret=LIVEKIT_API_SECRET,
                    ws_url=LIVEKIT_URL,
                    worker_type="room",
                    num_idle_processes=1,
                )
            )
        except Exception as e:
            print(f'❌ Error: {e}')
            print("\n💡 Sugerencia: Usa el modo directo:")
            print("   python voice_agent.py --direct")
            sys.exit(1)


if __name__ == "__main__":
    main()
