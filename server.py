# server.py - Servidor Flask para tokens de LiveKit

import os
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from livekit import api
from dotenv import load_dotenv
from datetime import timedelta
import json

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuración - IMPORTANTE: ws:// para WebSocket
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
LIVEKIT_ROOM = os.getenv('LIVEKIT_ROOM', 'test')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')

tool_data_storage = {}

print(f"🚀 LiveKit Server Config: {LIVEKIT_URL}")
print(f"🔑 API Key: {LIVEKIT_API_KEY[:10] if LIVEKIT_API_KEY else 'None'}...")


def create_token(room_name: str, participant_name: str,
                 can_publish: bool = True, can_subscribe: bool = True):
    """Crear token de acceso para un participante"""

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise ValueError("LIVEKIT_API_KEY y LIVEKIT_API_SECRET son requeridos")

    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.identity = participant_name
    token.name = participant_name

    # TTL como timedelta (10 horas)
    token.ttl = timedelta(hours=10)

    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=can_publish,
        can_subscribe=can_subscribe,
        can_publish_data=True,
    ))

    return token.to_jwt()


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'service': 'livekit-python-backend',
        'livekit_url': LIVEKIT_URL
    })


@app.route('/token', methods=['GET', 'POST'])
def get_token():
    """Generar token de acceso"""

    if request.method == 'POST':
        data = request.json
        room_name = data.get('room')
        participant_name = data.get('name')
    else:
        room_name = request.args.get('room')
        participant_name = request.args.get('name')

    if not room_name or not participant_name:
        return jsonify({
            'error': 'Se requieren parámetros "room" y "name"'
        }), 400

    try:
        token = create_token(room_name, participant_name)

        return jsonify({
            'token': token,
            'url': LIVEKIT_URL,  # ws:// para WebSocket
            'room': room_name,
            'identity': participant_name
        })
    except Exception as e:
        print(f"❌ Error generando token: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rooms', methods=['GET'])
def list_rooms():
    """Listar todas las salas activas - Versión sincrónica"""
    try:
        # Usar RoomServiceClient de forma síncrona
        room_service = api.RoomServiceClient(
            LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        )

        # Usar el método síncrono
        rooms_response = room_service.list_rooms(api.ListRoomsRequest())

        rooms_list = []
        for room in rooms_response.rooms:
            rooms_list.append({
                'name': room.name,
                'num_participants': room.num_participants,
                'max_participants': room.max_participants,
                'creation_time': str(room.creation_time) if room.creation_time else None,
            })

        return jsonify({
            'rooms': rooms_list
        })
    except Exception as e:
        print(f"❌ Error listando salas: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/create-room', methods=['POST'])
def create_room():
    """Crear una sala nueva"""
    try:
        room_service = api.RoomServiceClient(
            LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        )

        data = request.json or {}
        room_name = data.get('room', f'room-{int(asyncio.get_event_loop().time())}')

        room_service.create_room(api.CreateRoomRequest(
            name=room_name,
            empty_timeout=300,  # 5 minutos antes de cerrar sala vacía
            max_participants=10
        ))

        return jsonify({
            'status': 'ok',
            'room': room_name
        })
    except Exception as e:
        print(f"❌ Error creando sala: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Gestionar configuración de settings"""
    if request.method == 'POST':
        data = request.json or {}
        # Aquí puedes guardar la configuración en una base de datos o archivo
        print(f"📝 Configuración recibida: {data}")
        return jsonify({
            'status': 'ok',
            'message': 'Configuración guardada correctamente'
        })
    else:
        # Aquí puedes devolver la configuración actual
        return jsonify({
            'status': 'ok',
            'settings': {
                'livekit_url': LIVEKIT_URL,
                'room': LIVEKIT_ROOM
            }
        })


@app.route('/tool-data', methods=['POST'])
def tool_data():
    """Recibir datos de herramientas y enviarlos al agente vía Data Message"""
    try:
        data = request.json or {}
        room_name = data.get('room', LIVEKIT_ROOM)
        tool_name = data.get('tool_name', 'unknown')
        tool_data = data.get('data', {})
        user_identity = data.get('user_identity')

        print(f"🛠️ Datos de herramienta recibidos:")
        print(f"   Sala: {room_name}")
        print(f"   Herramienta: {tool_name}")
        print(f"   Usuario: {user_identity}")
        print(f"   Datos: {json.dumps(tool_data, indent=2)}")

        # Almacenar en memoria
        storage_key = f"{room_name}:{tool_name}"
        tool_data_storage[storage_key] = tool_data

        # Enviar data message al agente
        try:
            room_service = api.RoomServiceClient(
                LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
            )

            payload = json.dumps({
                'type': 'tool_data',
                'tool_name': tool_name,
                'data': tool_data,
                'user_identity': user_identity
            })

            room_service.send_data(api.SendDataRequest(
                room=room_name,
                data=payload.encode('utf-8'),
                kind=api.DataPacket.Kind.RELIABLE
            ))

            print(f"✅ Data message enviado al agente en sala '{room_name}'")

        except Exception as e:
            print(f"⚠️ No se pudo enviar data message: {e}")

        return jsonify({
            'status': 'ok',
            'message': 'Datos de herramientas procesados correctamente',
            'tool_name': tool_name,
            'storage_key': storage_key
        })

    except Exception as e:
        print(f"❌ Error procesando datos de herramientas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tool-data/<room_name>', methods=['GET'])
def get_tool_data(room_name):
    """🆕 ENDPOINT NUEVO - Consultar datos almacenados"""
    try:
        matching_data = {}
        prefix = f"{room_name}:"

        for key, value in tool_data_storage.items():
            if key.startswith(prefix):
                tool_name = key.replace(prefix, '')
                matching_data[tool_name] = value

        return jsonify({
            'status': 'ok',
            'room': room_name,
            'tools': matching_data
        })

    except Exception as e:
        print(f"❌ Error obteniendo datos: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    print(f"\n{'='*60}")
    print(f"🎥 LiveKit Python Backend - Token Server")
    print(f"{'='*60}")
    print(f"📡 Servidor HTTP: http://localhost:{port}")
    print(f"🔗 LiveKit WS: {LIVEKIT_URL}")
    print(f"🔑 API Key: {LIVEKIT_API_KEY[:15] if LIVEKIT_API_KEY else 'No configurada'}...")
    print(f"{'='*60}\n")
    print("Endpoints disponibles:")
    print(f"  GET  http://localhost:{port}/health          - Health check")
    print(f"  GET  http://localhost:{port}/token?room=XXX&name=YYY  - Obtener token")
    print(f"  GET  http://localhost:{port}/rooms           - Listar salas")
    print(f"  POST http://localhost:{port}/create-room     - Crear sala")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
