# AuraVoice - Backend (FastAPI)

El backend de AuraVoice está escrito en **Python 3.10+ con FastAPI**. Se encarga de procesar el audio, ejecutar la inferencia del modelo Wav2Vec2, gestionar las transmisiones de WebSocket para análisis en tiempo real y proveer las rutas necesarias para clasificar audios y reentrenar el modelo.

## 🏗 Arquitectura

El servidor utiliza la infraestructura orquestada con **Docker Compose**:
- **MySQL (`audio_training_db`)**: Base de datos relacional (ORM con SQLAlchemy) para guardar metadatos de las llamadas procesadas y el registro de los modelos entrenados.
- **MinIO**: Almacenamiento S3-compatible utilizado para guardar los archivos de audio binarios (`.wav`) persistidos tras las llamadas, y los artefactos del clasificador entrenado (`.joblib`).

## 📂 Estructura de Directorios

- `src/backend/api/routes/`: Definiciones de los endpoints expuestos (detección offline, websocket streaming, review/reentrenamiento).
- `src/backend/models/`: Declaraciones ORM de SQLAlchemy (`Audio` y `TrainedModel`).
- `src/backend/schemas/`: Validadores Pydantic que imponen los contratos de petición y respuesta (JSON payload).
- `src/backend/services/`: Capa lógica. Contiene el `AudioDetector` (puente con el clasificador y Wav2Vec2) y el `AudioNormalizer` para tratar los streams PCM entrantes.
- `src/backend/seeds/`: Scripts pre-configurados para crear el modelo base en la base de datos o poblar la tabla con audios ficticios.

## 🧠 Flujo del Modelo de Detección (Wav2Vec2)

El módulo principal en `services/detector.py` utiliza `transformers` (Meta) y `scikit-learn`. Toma señales de audio (normalizadas a 16kHz) y extrae características de la capa oculta 4 de Wav2Vec2. Esas características son introducidas al clasificador logístico para obtener la probabilidad final de ser voz sintética vs. humana (Deepfake).

## ⚡ Comandos Útiles

Asegúrate de estar utilizando el manejador de dependencias `uv` (y de preferencia, dentro del entorno virtual correspondiente) y correr los comandos desde la ruta raíz de `/backend/src`.

```bash
# Levantar el servicio
uv run uvicorn backend.main:app --reload
```

### Scripts Adicionales

1. **Seed del Modelo Base:** Obligatorio al correr el proyecto por primera vez para que el Review Hub tenga una referencia del modelo entrenado inicial.
   ```bash
   uv run python -m backend.seeds.seed_base_model
   ```
2. **Seed de Audios:** (Opcional) Introduce registros estáticos de llamadas a la base de datos para probar el frontend (Review Hub).
   ```bash
   uv run python -m backend.seed
   ```
3. **Validar Endpoint:** Ejecuta una petición batch `POST /detect` simulando el cliente exacto usado por el juez del Hackathon.
   ```bash
   uv run python backend/scripts/check_endpoint.py
   ```
