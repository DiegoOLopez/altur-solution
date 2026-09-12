# Backend de Altur Solution

API desarrollada con FastAPI para recibir y procesar audio. El proyecto usa `uv` para administrar el entorno virtual y las dependencias.

## Requisitos

- Python `3.14` o superior.
- [`uv`](https://docs.astral.sh/uv/) instalado.

Para instalar `uv` en macOS con Homebrew:

```bash
brew install uv
```

Para instalar `uv` en Windows con PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

También puedes instalarlo con `winget`:

```powershell
winget install --id=astral-sh.uv -e
```

## Instalación

Desde la carpeta `backend`:

```bash
uv sync
```

Este comando crea o actualiza el entorno virtual `.venv` e instala las dependencias definidas en `pyproject.toml` y `uv.lock`.

## Ejecutar el servidor

### Desarrollo

Desde `backend/`, ejecuta:

```bash
uv run uvicorn backend.main:app --reload
```

El servidor estará disponible en:

```text
http://127.0.0.1:8000
```

También puedes acceder a la documentación interactiva de FastAPI:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Configuración del host y puerto

Para permitir conexiones desde otros dispositivos de la red o cambiar el puerto:

```bash
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

### Estado de la API

```http
GET /
```

Respuesta:

```json
{
	"message": "Altur Solution API"
}
```

### Detección y normalización de audio

```http
POST /detect
```

Recibe un archivo WAV mediante `multipart/form-data`. El campo del formulario debe llamarse `file`.

Ejemplo con `curl`:

```bash
curl -X POST http://127.0.0.1:8000/detect \
	-F "file=@ruta/al/audio.wav"
```

El endpoint valida que el archivo tenga extensión `.wav`, lo procesa y, cuando es necesario, lo convierte de 8 kHz a 16 kHz.

Respuesta de ejemplo:

```json
{
	"message": "Audio converted from 8000 Hz to 16000 Hz.",
	"original_sample_rate": 8000,
	"final_sample_rate": 16000,
	"converted": true
}
```

### Revisión de audios

La revisión usa el modelo `Audio` y persiste los cambios en MySQL:

```http
GET /review/audios
GET /review/audios/{id}/stream
PATCH /review/audios/{id}/classification
DELETE /review/audios/{id}
GET /review/db/health
```

Para clasificar un audio, envía uno de estos valores:

```json
{
	"classification": "synthetic"
}
```

La clasificación `real` o `synthetic` marca el audio como revisado. `DELETE` hace un borrado lógico con estado `deleted`, por lo que no elimina el archivo físico. La reproducción busca cada `storage_key` dentro de `AUDIO_STORAGE_ROOT`; su valor predeterminado es `audio` y puede cambiarse mediante variable de entorno.

### Streaming por WebSocket

```text
WS /ws/detect
```

El servidor acepta mensajes de texto y responde con un JSON que incluye el contenido recibido.

Ejemplo conceptual de respuesta:

```json
{
	"message": "Detection WebSocket",
	"received": "mensaje"
}
```

## Estructura principal

```text
backend/
├── pyproject.toml
├── uv.lock
├── README.md
└── src/
		└── backend/
				├── main.py
				├── api/routes/
				├── core/
				├── models/
				├── schemas/
				└── services/
```

## Comandos útiles

Actualizar las dependencias:

```bash
uv lock
uv sync
```

Ejecutar el servidor sin activar manualmente el entorno virtual:

```bash
uv run uvicorn backend.main:app
```

Activar el entorno virtual manualmente, si se necesita:

```bash
source .venv/bin/activate
```
