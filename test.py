# test.py
import asyncio
from livekit import rtc
import sys

async def main():
    if len(sys.argv) < 2:
        print("Uso: python test.py <TOKEN>")
        sys.exit(1)

    token = sys.argv[1]
    url = "http://localhost:7880"  # ← CAMBIADO A HTTP

    room = rtc.Room()

    @room.on("participant_connected")
    def on_participant(participant):
        print(f"✅ {participant.identity} se unió")

    print("🔌 Conectando...")
    await room.connect(url, token)
    print(f"✅ Conectado a: {room.name}")
    print(f"👤 Tu identidad: {room.local_participant.identity}")
    print(f"\n👥 Participantes:")
    for p in room.remote_participants.values():
        print(f"   - {p.identity}")

    print("\n⏳ Ctrl+C para salir")
    await asyncio.sleep(float('inf'))

if __name__ == "__main__":
    asyncio.run(main())
