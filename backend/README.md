# Backend - Altur Solution

## Descripción
El backend de la solución Altur provee una API REST y un servicio WebSocket para la detección en tiempo real (y batch) de voces sintéticas en llamadas telefónicas. Es responsable de recibir el audio, interactuar con el modelo de IA para extraer características (acústicas y de comportamiento conversacional) y emitir un veredicto sobre la autenticidad de la voz. Además, proporciona funcionalidades para almacenar grabaciones en MinIO, registrar metadata en MySQL y ofrecer un panel de revisión para el reentrenamiento continuo (active learning) del modelo.

## Arquitectura

El flujo principal de procesamiento de audio es el siguiente:

```text
Cliente (Frontend / API REST / WebSocket)
  │
  ├─> API (FastAPI) recibe el audio WAV (estéreo)
  │
  ├─> Detector de Audio (AudioDetector)
  │     ├─> Resampling a 16 kHz (internamente)
  │     ├─> Separación de canales (Canal 0: Caller, Canal 1: Agente)
  │     ├─> Extracción de características acústicas (Caller)
  │     ├─> Análisis de comportamiento conversacional (Interacción Caller/Agente)
  │     └─> Stacking Logístico (LLR Acústico + LLR Comportamental)
  │
  └─> Veredicto Final (is_synthetic, confidence, scores)
```

## Tecnologías

- **Python 3.14+**
- **FastAPI**: Framework web principal.
- **Uvicorn**: Servidor ASGI.
- **SQLAlchemy & Alembic**: ORM y migraciones de base de datos MySQL.
- **MinIO (minio-py)**: Almacenamiento de objetos S3 compatible para archivos `.wav` y `.pkl`.
- **Joblib, NumPy, SciPy, scikit-learn**: Carga de modelos de IA (Pipeline de Densidades y Stacking) y procesamiento matricial.
- **Librosa & Soundfile**: Procesamiento y resampling de audio.
- **Silero VAD / SpeechBrain**: Detección de actividad de voz y embeddings.

## Requisitos

- Python 3.14 o superior.
- [uv](https://github.com/astral-sh/uv) (Gestor de paquetes y dependencias).
- Docker y Docker Compose (para levantar MySQL y MinIO localmente).

## Instalación

1. **Clonar y acceder al directorio del backend:**
   ```bash
   cd backend
   ```

2. **Levantar los servicios de infraestructura (MySQL y MinIO):**
   ```bash
   docker-compose up -d
   ```

3. **Instalar dependencias utilizando `uv`:**
   ```bash
   uv sync
   ```
   *Esto creará automáticamente el entorno virtual `.venv` y descargará las dependencias definidas en `pyproject.toml`.*

4. **Ejecutar migraciones de la Base de Datos:**
   ```bash
   uv run alembic upgrade head
   ```

5. **Sembrar el modelo base en la Base de Datos:**
   ```bash
   uv run python -m src.backend.seeds.seed_base_model
   ```

## Variables de entorno

El proyecto no utiliza un archivo `.env` por defecto, ya que los valores de conexión se encuentran configurados para un entorno local/hackathon en el código fuente (ej. `core/database.py`, `core/config.py`). Sin embargo, se pueden sobrescribir mediante variables de entorno gracias a `pydantic-settings`.

Las variables principales reconocidas por `Settings` son:

- `APP_NAME` (Opcional): Nombre de la aplicación (Por defecto: `"Altur Solution API"`).
- `APP_VERSION` (Opcional): Versión de la API (Por defecto: `"0.1.0"`).
- `AUDIO_STORAGE_ROOT` (Opcional): Raíz para almacenamiento (Por defecto: `"audio"`).

**Credenciales integradas (Hardcoded para entorno local):**
- MySQL URL: `mysql+pymysql://api_user:api_password@127.0.0.1:3306/audio_training_db`
- MinIO URL: `localhost:9000` (User: `admin`, Pass: `supersecretpassword`)

## Ejecución

Para iniciar el servidor de desarrollo en local:

```bash
uv run uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Una vez en ejecución, puedes acceder a la documentación interactiva Swagger (OpenAPI) visitando:
👉 `http://localhost:8000/docs`

## API

### `POST /detect`

Analiza un clip de audio de una llamada telefónica para determinar si la voz es sintética o humana. 

**Nota:** Aunque en otros sistemas el audio puede recibirse en Base64, en esta implementación la API requiere que el archivo sea subido mediante `multipart/form-data`. El modelo espera un archivo `.wav` que incluya la conversación estéreo (Canal 0 = Cliente/Caller, Canal 1 = Agente) para poder extraer características de comportamiento.

- **Método:** `POST`
- **Endpoint:** `/detect`
- **Content-Type:** `multipart/form-data`
- **Formato del audio:** WAV (estéreo recomendado). El modelo resamplea internamente a 16 kHz. 

#### Request
Ejemplo usando `curl`:

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ruta/al/audio/llamada.wav"
```

#### Response (200 OK)

El modelo devuelve un JSON detallado con el veredicto y las métricas acumuladas (LLR).

```json
{
  "is_synthetic": false,
  "confidence": 0.8924,
  "score_total": -3.456,
  "llr_acoustic_cum": -2.120,
  "llr_behavioral_cum": -1.336,
  "n_acoustic_segments": 14,
  "n_behavioral_events": 5,
  "eta": 0.0
}
```

#### Posibles errores HTTP
- **400 Bad Request:** 
  - `No filename provided.` (El archivo no tiene nombre).
  - `Only WAV files are supported.` (La extensión no es `.wav`).
  - `The uploaded file is empty.` (Archivo vacío).
  - Errores generados internamente por el pipeline acústico.
- **422 Unprocessable Entity:** Falla en la validación de los datos (Pydantic).
- **500 Internal Server Error:** `Detection failed: <mensaje_error>` (Fallo general del modelo o pipeline).

### Otros Endpoints Relevantes

El backend también expone endpoints para el módulo de reentrenamiento continuo (Active Learning):

- `GET /review/audios`: Lista de grabaciones pendientes de revisión.
- `GET /review/stats`: Estadísticas de revisión y reentrenamiento.
- `GET /review/audios/{id}/stream`: Stream del audio directamente desde MinIO para el reproductor.
- `PATCH /review/audios/{id}/classification`: Etiquetar un audio manualmente (`synthetic` o `real`).
- `POST /review/train`: Reentrena un modelo nuevo desde cero fusionando el modelo actual con las nuevas etiquetas manuales, guarda el binario `.pkl` en MinIO y registra las métricas en MySQL.
- `WS /ws/call`: Conexión WebSocket para el streaming en vivo (utilizado en el Simulador).
