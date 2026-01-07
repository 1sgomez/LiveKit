# 🎯 Voice Agent Integrado - LiveKit

Sistema completo de agente de voz con integración backend-frontend y data tools dinámicos.

## 📋 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA INTEGRADO                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Frontend   │─────▶│   Backend    │─────▶│  Agente  │ │
│  │   (HTML/JS)  │      │  (Flask API) │      │  (Voz)   │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         │                      │                      │      │
│         │                      │                      │      │
│         └──────────┬───────────┴──────────┬───────────┘      │
│                    │                      │                 │
│         ┌──────────▼──────────┐  ┌────────▼─────────┐      │
│         │  Data Tools         │  │ LiveKit Server   │      │
│         │  (Contexto)         │  │  (WebRTC/WS)     │      │
│         └─────────────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Instalación Rápida

### 1. Requisitos Previos

```bash
# Python 3.8+
# Ollama instalado y corriendo
# LiveKit Server instalado y corriendo
```

### 2. Instalar Dependencias

```bash
pip install flask flask-cors livekit requests numpy librosa soundfile pydub gtts whisper
```

### 3. Configurar Entorno

```bash
# Crear archivo .env
cat > .env << EOF
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=tu_api_key
LIVEKIT_API_SECRET=tu_api_secret
LIVEKIT_ROOM=test
FLASK_PORT=5000
EOF
```

### 4. Iniciar Servicios

```bash
# Terminal 1: LiveKit Server
livekit-server

# Terminal 2: Ollama
ollama serve

# Opcional: Descargar modelo
ollama pull gemma3:1b
```

## 🏃‍♂️ Ejecución

### Opción 1: Script Automático (Recomendado)

```bash
python setup_and_run.py
# Selecciona opción 1 (Todo en uno)
```

### Opción 2: Manual

```bash
# Terminal 1: Backend
python server_integrated.py

# Terminal 2: Agente
python voice_agent_integrated.py --direct

# Navegador: http://localhost:5000/
```

## 📁 Estructura de Archivos

```
├── server_integrated.py      # Backend Flask con data tools
├── voice_agent_integrated.py # Agente con contexto dinámico
├── setup_and_run.py          # Script de instalación y ejecución
├── demo_integration.py       # Demostración de integración
├── settings_config.py        # Configuraciones del proyecto
├── README_INTEGRADO.md       # Documentación
├── .env                      # Variables de entorno
└── voice_agent.html          # Frontend (legacy)
```

## 🛠️ Componentes Principales

### 1. Backend Integrado (`server_integrated.py`)

**Funcionalidades:**

- Generación de tokens de LiveKit
- Manejo de sesiones de usuarios
- Almacenamiento de data tools
- API REST para frontend
- Frontend HTML integrado

**Endpoints Clave:**

```http
GET  /health                    # Estado del sistema
GET  /settings                  # Configuraciones
POST /session/start            # Iniciar sesión con data tools
GET  /session/{id}/data-tools  # Obtener data tools
POST /token                    # Token legacy
```

### 2. Agente Integrado (`voice_agent_integrated.py`)

**Nuevas Características:**

- Recepción de data tools vía WebSocket
- Contexto de sesión por usuario
- Análisis automático de herramientas
- Integración con LLM para respuestas contextuales

**Flujo de Procesamiento:**

1. Recibe audio del usuario
2. Transcribe con Whisper
3. Obtiene data tools de la sesión
4. Genera respuesta con contexto
5. Sintetiza voz con gTTS
6. Publica audio en LiveKit

### 3. Frontend Automático

**Características:**

- Conexión automática al cargar página
- Envío de data tools al backend
- UI moderna con logs en tiempo real
- Temas personalizables

## 🎯 Data Tools - Contexto Dinámico

### Qué son los Data Tools?

Información contextual que el agente usa para personalizar respuestas.

### Ejemplo de Estructura:

```json
{
  "user_context": {
    "name": "Juan Pérez",
    "role": "Desarrollador",
    "preferences": ["Python", "Automatización"]
  },
  "app_data": {
    "version": "2.0.0",
    "features": ["voice", "tools"]
  },
  "tour_data": {
    "steps": [{ "element": "#main", "title": "Bienvenida" }]
  }
}
```

### Cómo se Envían?

1. **Frontend** → Obtiene data tools (desde tu API, BD, etc.)
2. **Frontend** → Inicia sesión con data tools
3. **Backend** → Almacena en sesión
4. **Agente** → Recibe vía WebSocket
5. **Agente** → Usa en generación de respuestas

### Ejemplos de Uso:

**Usuario:** "¿Qué sabes de mi contexto?"
**Agente:** "Sé que eres Juan Pérez, desarrollador, te gusta Python y la automatización."

**Usuario:** "¿Qué funciones tiene la app?"
**Agente:** "Tu app v2.0.0 tiene funciones de voz, herramientas y contexto inteligente."

**Usuario:** "¿Qué pasos del tour recuerdo?"
**Agente:** "Tienes un tour con 3 pasos: Bienvenida, Configuración y Controles."

## ⚙️ Configuración (`settings_config.py`)

### Personalización:

```python
settings = VoiceAgentSettings(
    project_name="Mi Proyecto",
    llm_model="llama3.2:1b",
    language="es",
    auto_connect=True,
    enable_tools=True,
    # ... más opciones
)
```

### Variables de Entorno:

```bash
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
LIVEKIT_ROOM=project-room
FLASK_PORT=5000
```

## 🔌 Integración con Tu Backend

### Paso 1: Modificar `server_integrated.py`

```python
@app.route('/get-user-data', methods=['POST'])
def get_user_data():
    user_id = request.json.get('user_id')

    # Obtener datos de tu base de datos/API
    data_tools = {
        "user_context": get_user_from_db(user_id),
        "app_data": get_app_config(),
        "business_data": get_business_context()
    }

    return jsonify(data_tools)
```

### Paso 2: Frontend obtiene data tools

```javascript
async function getDataTools(userId) {
  const response = await fetch("/get-user-data", {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
  });
  return await response.json();
}
```

### Paso 3: Iniciar sesión con data tools

```javascript
const dataTools = await getDataTools("user123");
const response = await fetch("http://localhost:5000/session/start", {
  method: "POST",
  body: JSON.stringify({
    user_id: "user123",
    data_tools: dataTools,
  }),
});
```

## 🎨 Temas del Frontend

### Modern (por defecto)

- Gradiente azul/púrpura
- UI rica con logs
- Botones grandes

### Minimal

- Interfaz limpia
- Menos elementos
- Enfoque en funcionalidad

### Dark

- Tema oscuro
- Alto contraste
- Ideal para desarrollo

**Cambiar tema:** Editar `frontend_theme` en `settings_config.py`

## 📊 Monitoreo y Logs

### Backend:

```bash
# Logs en tiempo real
tail -f logs/backend.log
```

### Agente:

```bash
# Ver procesos
ps aux | grep voice_agent
```

### Frontend:

- Logs visibles en la interfaz
- Consola del navegador (F12)

## 🔧 Troubleshooting

### Problema: "Ollama no disponible"

```bash
# Solución
ollama serve
# O en otro terminal
ollama pull gemma3:1b
```

### Problema: "LiveKit no conecta"

```bash
# Verificar servidor
livekit-server --help
# O verificar puerto
netstat -tulpn | grep 7880
```

### Problema: "Dependencias faltantes"

```bash
# Instalar todas
pip install flask flask-cors livekit requests numpy librosa soundfile pydub gtts whisper
```

### Problema: "Puerto ocupado"

```bash
# Cambiar puerto en .env
FLASK_PORT=5001
```

## 🚀 Características Avanzadas

### 1. Barge-in (Interrupción)

El agente puede ser interrumpido hablando por encima.

### 2. Contexto Persistente

La conversación mantiene contexto entre turnos.

### 3. Multi-usuario

Cada usuario tiene su propio contexto y sesión.

### 4. Data Tools en Tiempo Real

Puedes actualizar data tools durante la sesión.

### 5. Auto-reconexión

El frontend intenta reconectar automáticamente.

## 📈 Rendimiento

### Optimizaciones:

- **Latencia baja**: Chunk size 1024, sleep optimizado
- **Pre-roll buffer**: Captura 300ms antes del habla
- **Cache de contexto**: Evita recálculos
- **Stream processing**: Audio procesado en tiempo real

### Benchmarks:

- Transcripción: ~200ms (Whisper small)
- Generación: ~500ms (Ollama gemma3:1b)
- Síntesis: ~300ms (gTTS)
- **Total**: ~1s (latencia end-to-end)

## 🛡️ Seguridad

### Recomendaciones:

1. Usa API keys reales en producción
2. Implementa autenticación de usuarios
3. Valida data tools entrantes
4. Rate limiting en endpoints
5. HTTPS en producción

### Variables sensibles:

```bash
# Nunca commitear .env
echo ".env" >> .gitignore
```

## 📚 Ejemplos de Uso

### Ejemplo 1: Soporte Técnico

```json
{
  "user_context": {
    "name": "Ana",
    "role": "Soporte",
    "tickets": ["#1234", "#5678"]
  },
  "app_data": {
    "version": "1.5.2",
    "issues": ["login", "payment"]
  }
}
```

**Pregunta:** "¿Qué problemas tengo pendientes?"
**Respuesta:** "Tienes 2 tickets pendientes: #1234 (login) y #5678 (payment)."

### Ejemplo 2: E-commerce

```json
{
  "user_context": {
    "name": "Carlos",
    "preferences": ["tecnología", "gadgets"],
    "cart": ["laptop", "mouse"]
  },
  "cart_data": {
    "items": 2,
    "total": 1200
  }
}
```

**Pregunta:** "¿Qué tengo en el carrito?"
**Respuesta:** "Tienes 2 artículos: laptop y mouse, total $1200."

### Ejemplo 3: Onboarding

```json
{
  "tour_data": {
    "steps": [
      { "title": "Inicio", "completed": true },
      { "title": "Configuración", "completed": false },
      { "title": "Tutorial", "completed": false }
    ]
  }
}
```

**Pregunta:** "¿Qué falta completar?"
**Respuesta:** "Te falta completar: Configuración y Tutorial."

## 🤝 Contribución

### Estructura de código:

- Usa type hints
- Documenta funciones
- Sigue PEP 8
- Tests para funciones críticas

### Extensión:

```python
# Añadir nueva herramienta
async def analyze_custom_data(self, data: dict) -> str:
    # Tu lógica aquí
    return "Análisis personalizado"

# En generate_response:
if "custom" in user_message.lower():
    tool_response = await self.analyze_custom_data(data_tools)
```

## 📄 Licencia

MIT License - Usa libremente en proyectos personales y comerciales.

---

**Desarrollado con ❤️ para integración backend-frontend de voz**

**Versión:** 2.0.0
**Documentación:** Actualizada 2026
