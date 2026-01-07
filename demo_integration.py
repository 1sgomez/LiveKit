#!/usr/bin/env python3
# demo_integration.py - Demostración de integración completa

import requests
import json
import time
import sys

def demo_backend_integration():
    """Demostrar la integración completa del backend"""

    BASE_URL = "http://localhost:5000"

    print("="*70)
    print("🎯 DEMO: Integración Backend + Frontend + Data Tools")
    print("="*70)

    # 1. Verificar salud del backend
    print("\n1. Verificando backend...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Backend activo: {data['service']}")
            print(f"   🔗 LiveKit URL: {data['livekit_url']}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Backend no disponible: {e}")
        print("   💡 Asegúrate de ejecutar: python server_integrated.py")
        return False

    # 2. Verificar configuraciones
    print("\n2. Consultando configuraciones...")
    try:
        response = requests.get(f"{BASE_URL}/settings")
        settings = response.json()
        print(f"   ⚙️  Settings: {json.dumps(settings, indent=2)}")
    except Exception as e:
        print(f"   ⚠️  No se pudo obtener settings: {e}")

    # 3. Crear data tools para demo
    print("\n3. Creando data tools de demostración...")
    demo_data_tools = {
        "user_context": {
            "name": "Juan Pérez",
            "role": "Desarrollador",
            "preferences": ["tecnología", "automatización", "python"],
            "session_type": "soporte_técnico"
        },
        "app_data": {
            "version": "2.1.0",
            "features": ["voice", "tools", "auto-connect", "context-aware"],
            "environment": "production",
            "project": "LiveKit Voice Agent"
        },
        "tour_data": {
            "steps": [
                {
                    "element": "#main-dashboard",
                    "title": "Panel Principal",
                    "description": "Acceso a todas las funciones de voz"
                },
                {
                    "element": "#settings-panel",
                    "title": "Configuración",
                    "description": "Personaliza tu experiencia"
                },
                {
                    "element": "#voice-controls",
                    "title": "Controles de Voz",
                    "description": "Activa/desactiva el micrófono"
                }
            ]
        },
        "api_context": {
            "endpoints": ["/api/users", "/api/settings", "/api/voice"],
            "rate_limit": 100,
            "timeout": 30
        }
    }

    print(f"   🛠️  Data tools creados:")
    for tool_name in demo_data_tools.keys():
        print(f"      - {tool_name}")

    # 4. Iniciar sesión con data tools
    print("\n4. Iniciando sesión con data tools...")
    try:
        response = requests.post(
            f"{BASE_URL}/session/start",
            json={
                "user_id": "demo_user_001",
                "room_name": f"demo-room-{int(time.time())}",
                "data_tools": demo_data_tools
            },
            timeout=10
        )

        if response.status_code == 200:
            session_data = response.json()
            print(f"   ✅ Sesión iniciada!")
            print(f"      Session ID: {session_data['session_id']}")
            print(f"      Room: {session_data['room']}")
            print(f"      Token: {session_data['token'][:50]}...")

            # Guardar para uso posterior
            session_id = session_data['session_id']
            room_name = session_data['room']
            token = session_data['token']
            url = session_data['url']

        else:
            print(f"   ❌ Error iniciando sesión: {response.status_code}")
            print(f"      {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # 5. Verificar data tools almacenados
    print("\n5. Verificando data tools en sesión...")
    try:
        response = requests.get(f"{BASE_URL}/session/{session_id}/data-tools")
        if response.status_code == 200:
            stored_tools = response.json()
            print(f"   ✅ Data tools almacenados: {list(stored_tools['data_tools'].keys())}")
        else:
            print(f"   ⚠️  No se pudo verificar: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error verificando: {e}")

    # 6. Actualizar estado de sesión
    print("\n6. Actualizando estado de sesión...")
    try:
        response = requests.put(
            f"{BASE_URL}/session/{session_id}/status",
            json={"status": "active"}
        )
        if response.status_code == 200:
            print("   ✅ Estado actualizado a 'active'")
        else:
            print(f"   ⚠️  Error: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    # 7. Listar salas activas
    print("\n7. Consultando salas activas...")
    try:
        response = requests.get(f"{BASE_URL}/rooms")
        if response.status_code == 200:
            rooms_data = response.json()
            rooms = rooms_data.get('rooms', [])
            print(f"   📊 Salas activas: {len(rooms)}")
            for room in rooms:
                print(f"      - {room['name']}: {room['num_participants']} participantes")
        else:
            print(f"   ⚠️  Error: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    # 8. Demostrar cómo usar el frontend
    print("\n8. INSTRUCCIONES PARA EL FRONTEND:")
    print(f"   🌐 Accede a: http://localhost:5000/")
    print(f"   🧪 Demo rápida: http://localhost:5000/demo")
    print(f"   📋 Session ID para referencia: {session_id}")

    print("\n9. CÓMO USAR EL SISTEMA:")
    print("   a) Ejecuta el backend: python server_integrated.py")
    print("   b) Ejecuta el agente: python voice_agent_integrated.py --direct")
    print("   c) Abre el navegador: http://localhost:5000/")
    print("   d) El frontend se conectará automáticamente")
    print("   e) El agente recibirá los data tools")
    print("   f) Habla con el agente - usará el contexto!")

    print("\n10. EJEMPLO DE FLUJO:")
    print("    Usuario: '¿Qué sabes de mi contexto?'")
    print("    Agente: 'Sé que eres Juan Pérez, desarrollador,")
    print("            te gusta la tecnología y Python. Estás en modo soporte técnico.'")

    print("\n" + "="*70)
    print("✅ Demo completada!")
    print("="*70)

    return True

def demo_frontend_scenario():
    """Demostrar escenario de frontend"""
    print("\n" + "="*70)
    print("🌐 ESCENARIO FRONTEND")
    print("="*70)

    print("""
Cuando un usuario entra a la página:

1. El frontend carga automáticamente
2. Obtiene data tools desde tu API/backend
3. Inicia sesión con esos data tools
4. El agente recibe el contexto
5. El usuario habla y el agente usa el contexto

EJEMPLO DE DATA TOOLS QUE PUEDES INTEGRAR:
- Datos del usuario (nombre, preferencias)
- Contexto de la aplicación (versión, features)
- Tour/Onboarding (pasos del usuario)
- Datos de negocio (productos, historial)
- Configuraciones (idioma, zona horaria)
""")

if __name__ == "__main__":
    # Verificar si el servidor está corriendo
    if demo_backend_integration():
        demo_frontend_scenario()
    else:
        print("\n❌ No se pudo completar la demo")
        sys.exit(1)
