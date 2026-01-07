# voice_agent.py - Agente de voz 100% local con LiveKit
# Uso: python voice_agent.py --direct

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
import soundfile as sf
import tempfile
from pydub import AudioSegment
import aiohttp
from collections import deque
import librosa

# Importaciones de LiveKit
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)

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

class SimpleVAD:
    """Detector de actividad de voz simple basado en energía"""

    def __init__(self, threshold: float = 0.02, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate

    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Detectar si hay voz basada en energía"""
        # Calcular RMS (Root Mean Square) - medida de volumen
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > self.threshold

class LocalVoiceAgent:
    def __init__(self, room_name: str, use_simple_vad: bool = False):
        # Buffer de pre-roll (~300 ms)
        self.preroll_buffer = deque(maxlen=10)

        self.room_name = room_name
        self.ctx_room = None
        self.conversation_history = []
        self.audio_buffer = []
        self.is_processing = False
        self.silence_detected = False
        self.greeting_sent = False
        self.use_simple_vad = use_simple_vad

        # Cargar modelos
        logger.info("📥 Cargando Whisper (STT)...")
        self.whisper_model = whisper.load_model("small")

        # Intentar cargar Silero VAD
        self.vad = None
        self.simple_vad = SimpleVAD(threshold=0.004, sample_rate=16000)

        try:
            from livekit.plugins import silero
            logger.info("📥 Cargando Silero VAD...")
            self.vad = silero.VAD.load()
            logger.info("✅ Silero VAD cargado")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar Silero VAD: {e}")
            logger.info("   Usando detector de volumen simple")

        logger.info("✅ Modelos cargados")

    async def connect(self, url: str, token: str):
        """Conectar directamente a una sala"""
        logger.info(f"🔌 Conectando a la sala {self.room_name}...")

        self.ctx_room = rtc.Room()

        self.ctx_room.on("track_subscribed", self.on_track_subscribed)
        self.ctx_room.on("participant_connected", self.on_participant_connected)
        self.ctx_room.on("connected", self.on_connected)

        await self.ctx_room.connect(url, token)

        logger.info(f"✅ Conectado a {self.room_name}")
        logger.info(f"👤 Identity: {self.ctx_room.local_participant.identity}")

    def detect_speech(self, audio_data: np.ndarray) -> bool:
        """Detectar si hay voz"""
        if self.vad is not None:
            try:
                # Intentar API de Silero VAD
                if hasattr(self.vad, 'is_speech'):
                    return self.vad.is_speech(audio_data, sample_rate=16000)
                elif callable(self.vad):
                    # Nueva API: VAD es un callable
                    return self.vad(audio_data, sample_rate=16000) > 0.5
            except Exception as e:
                logger.debug(f"VAD error, usando fallback: {e}")

        # Fallback: usar detector simple por volumen
        return self.simple_vad.is_speech(audio_data)

    async def transcribe_audio(self, audio_data: bytes, input_sample_rate: int) -> str:
        try:
            # PCM int16 → float32
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            # Resample REAL → 16 kHz (OBLIGATORIO)
            if input_sample_rate != 16000:
                audio_np = librosa.resample(
                    audio_np,
                    orig_sr=input_sample_rate,
                    target_sr=16000
                )

            # Guardar WAV correcto
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
                sf.write(wav_path, audio_np, 16000, subtype="PCM_16")

            result = self.whisper_model.transcribe(
                wav_path,
                language="es",
                task="transcribe",
                initial_prompt="Conversación en español. El usuario habla con un asistente llamado Jarvis.",
                temperature=0.0,
                fp16=False
            )

            os.unlink(wav_path)

            text = result["text"].strip()
            if text:
                logger.info(f"👤 Usuario: {text}")

            return text

        except Exception as e:
            logger.error(f"❌ Error en transcripción: {e}")
            return ""


    async def generate_response(self, user_message: str) -> str:
        try:
            self.conversation_history.append(
                {"role": "user", "content": user_message}
            )

            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2:3b",
                            "options": {
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "num_predict": 60,
                            "repeat_penalty": 1.2
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content":
                                "Eres Jarvis, un asistente de voz."
                                "Respondes SOLO en español."
                                "Tus respuestas son cortas (máx 2 frases)."
                                "Hablas de forma natural y directa."
                                "No explicas limitaciones técnicas."
                                "No mencionas que eres un modelo de lenguaje."
                                "Si no tienes un dato, dilo de forma breve y humana."
                            },
                            *self.conversation_history
                        ],
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:

                    result = await resp.json()
                    assistant_message = result["message"]["content"].strip()

                    # MAX_WORDS = 35
                    # words = assistant_message.split()

                    # if len(words) > MAX_WORDS:
                    #     assistant_message = "Claro. ¿Qué necesitas saber exactamente?"

                    self.conversation_history.append(
                        {"role": "assistant", "content": assistant_message}
                    )

                    logger.info(f"🤖 Jarvis: {assistant_message}")
                    return assistant_message

        except Exception as e:
            logger.error(f"❌ Error con Ollama (async): {e}")
            return "No puedo responder en este momento."

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        try:
            tts = gTTS(text=text, lang='es')

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                mp3_path = f.name
                tts.save(mp3_path)

            audio = AudioSegment.from_mp3(mp3_path).set_frame_rate(16000).set_channels(1)
            os.unlink(mp3_path)

            return audio.raw_data

        except Exception as e:
            logger.error(f"❌ Error TTS: {e}")
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
        if self.is_processing:
            return

        self.is_processing = True
        self.audio_buffer = []
        self.preroll_buffer.clear()
        self.silence_detected = False
        silence_start = None
        speech_detected = False
        input_sample_rate = None

        logger.info(f"🎤 Escuchando a {participant.identity}...")

        audio_stream = rtc.AudioStream(track)
        silence_timeout = 1  # segundos

        try:
            async for frame_event in audio_stream:
                current_time = asyncio.get_event_loop().time()

                input_sample_rate = frame_event.frame.sample_rate

                audio_data = (
                    np.frombuffer(frame_event.frame.data, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )

                # Guardar siempre pre-roll
                self.preroll_buffer.append(frame_event.frame.data)

                is_speech = self.detect_speech(audio_data)

                if is_speech:
                    if not speech_detected:
                        # Primer frame de voz → agregar pre-roll
                        self.audio_buffer.extend(self.preroll_buffer)
                        logger.debug("🎙️ Pre-roll agregado")

                    self.audio_buffer.append(frame_event.frame.data)
                    self.silence_detected = False
                    silence_start = None
                    speech_detected = True

                else:
                    if speech_detected:
                        if not self.silence_detected:
                            self.silence_detected = True
                            silence_start = current_time
                        elif silence_start and (current_time - silence_start) >= silence_timeout:
                            logger.info(
                                f"🛑 Fin del habla ({current_time - silence_start:.2f}s silencio)"
                            )
                            break

            if self.audio_buffer:
                logger.info(f"📝 Procesando {len(self.audio_buffer)} frames de audio...")

                audio_bytes = b"".join(self.audio_buffer)

                text = await self.transcribe_audio(audio_bytes, input_sample_rate)

                if text:
                    response = await self.generate_response(text)
                    audio_response = await self.synthesize_speech(response)
                    if audio_response:
                        await self.publish_audio(audio_response)
                else:
                    logger.info("🤷 Whisper no detectó texto")

        except Exception as e:
            logger.error(f"❌ Error procesando audio: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_processing = False
            self.audio_buffer = []

    async def listen_loop(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        while True:
            try:
                await self.handle_audio_stream(track, participant)
                await asyncio.sleep(0.3)  # pequeño respiro entre turnos
            except Exception as e:
                logger.error(f"❌ Error en loop de escucha: {e}")
                await asyncio.sleep(1)

    def on_track_subscribed(self, track: rtc.Track, publication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"🎧 Audio de {participant.identity}")
            asyncio.ensure_future(self.listen_loop(track, participant))

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

        ctx.room.on("track_subscribed", agent.on_track_subscribed)
        ctx.room.on("participant_connected", agent.on_participant_connected)
        ctx.room.on("connected", agent.on_connected)

        logger.info(f"📍 Conectado a: {ctx.room.name}")
        await asyncio.sleep(float('inf'))

    except Exception as e:
        logger.error(f"❌ Error: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Agente de voz Jarvis")
    parser.add_argument("--direct", action="store_true",
                       help="Modo directo: conectar directamente a la sala")
    parser.add_argument("--simple-vad", action="store_true",
                       help="Usar detector de volumen simple en lugar de Silero VAD")
    args = parser.parse_args()

    print('='*60)
    print('🚀 Voice Agent - Jarvis')
    print('='*60)

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"   LIVEKIT_URL: {LIVEKIT_URL}")
    print(f"   LIVEKIT_ROOM: {LIVEKIT_ROOM}")
    print()

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("❌ ERROR: Faltan LIVEKIT_API_KEY o LIVEKIT_API_SECRET")
        sys.exit(1)

    if args.direct:
        print("🎯 MODO: Directo")
        print("   El agente se conectará inmediatamente a la sala")
        print()

        try:
            asyncio.run(run_direct_mode())
        except KeyboardInterrupt:
            print("\n👋 Agente detenido")
    else:
        print("🎯 MODO: Worker")
        print("   El agente esperará trabajos de LiveKit")
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
