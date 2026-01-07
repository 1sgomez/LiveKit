# Agente de Voz Local

Un agente de voz 100% local que utiliza Whisper para STT, Ollama para LLM y gTTS para TTS.

## Requisitos

- Python 3.8+
- LiveKit Server
- Ollama con modelo llama3.2:3b
- Conexión a Internet para gTTS (o configurar TTS local)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/voice-agent.git
cd voice-agent

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Copiar archivo de entorno
cp .env.example .env

# Editar .env con tus credenciales de LiveKit
```

## Configuración

1. **LiveKit**: Necesitas un servidor LiveKit en ejecución. Puedes usar el local o un servicio en la nube.
2. **Ollama**: Debes tener Ollama instalado y el modelo `llama3.2:3b` descargado:
   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```
3. **Variables de entorno**: Configura `.env` con tus credenciales.

## Ejecución

```bash
# Iniciar el agente
python voice_agent.py

# Para especificar una sala diferente
LIVEKIT_ROOM=mi-sala python voice_agent.py
```

## Solución de problemas

### El agente se queda "iniciando"

Si el agente solo muestra logs de `name main` y no avanza:

1. **Verifica que LiveKit esté en ejecución**:

   ```bash
   docker ps  # Si usas Docker
   # o
   livekit-server --help
   ```

2. **Verifica las credenciales**: Asegúrate de que `LIVEKIT_API_KEY` y `LIVEKIT_API_SECRET` sean correctos.

3. **Verifica la conexión a la sala**: El agente necesita unirse a una sala específica. Puedes configurar el nombre de la sala con `LIVEKIT_ROOM`.

4. **Verifica que Ollama esté en ejecución**:

   ```bash
   curl http://localhost:11434/api/tags
   ```

5. **Verifica los logs**: Ejecuta con más verbosidad:
   ```bash
   LIVEKIT_LOG_LEVEL=debug python voice_agent.py
   ```

### No se detecta audio

1. Verifica que tu micrófono esté funcionando
2. Asegúrate de que el navegador tenga permisos para acceder al micrófono
3. Verifica que estés conectado a la misma sala que el agente

## Arquitectura

```
Usuario → Micrófono → LiveKit → Agente → Whisper (STT) → Ollama (LLM) → gTTS (TTS) → LiveKit → Altavoz → Usuario
```

## Dependencias principales

- `livekit`: Para la conexión WebRTC
- `whisper`: Para reconocimiento de voz
- `gTTS`: Para síntesis de voz
- `ollama`: Para el modelo de lenguaje
- `silero`: Para detección de actividad de voz (VAD)

## Licencia

MIT
