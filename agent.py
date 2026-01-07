# agent.py
import asyncio
import logging
import os
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Leer configuración del .env
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')

logger.info(f"🔧 Configuración del agente:")
logger.info(f"   URL: {LIVEKIT_URL}")
logger.info(f"   API Key: {LIVEKIT_API_KEY}")
logger.info(f"   API Secret: {'*' * len(LIVEKIT_API_SECRET) if LIVEKIT_API_SECRET else 'NO CONFIGURADO'}")


class SimpleVoiceAgent:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.room = ctx.room
        self.participant_count = 0

    async def run(self):
        """Lógica principal del agente"""
        logger.info(f"🤖 Agente iniciado en sala: {self.room.name}")

        # Evento: Participante conectado
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            self.participant_count += 1
            logger.info(f"👤 Participante #{self.participant_count} conectado: {participant.identity}")

        # Evento: Participante desconectado
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            self.participant_count -= 1
            logger.info(f"👋 Participante desconectado: {participant.identity}")

        # Evento: Track recibido (audio/video)
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            logger.info(f"📡 Track recibido de {participant.identity}")
            logger.info(f"   Tipo: {track.kind}")

            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info("🎤 Audio detectado - Procesando voz")

        # Mantener agente activo
        logger.info("⏳ Agente listo y esperando...")
        await asyncio.sleep(float('inf'))


async def entrypoint(ctx: JobContext):
    """Punto de entrada del agente"""
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"✅ Agente conectado a sala: {ctx.room.name}")

    agent = SimpleVoiceAgent(ctx)
    await agent.run()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            ws_url=LIVEKIT_URL.replace('http://', 'ws://'),  # Convertir http a ws
        )
    )
