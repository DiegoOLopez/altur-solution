"""
AuraVoice — Punto de entrada de la API.

Levanta la aplicación FastAPI que sirve los tres flujos de Altur:

- ``POST /detect``      : análisis forense de una llamada completa (batch).
- ``POST /detect_wav``  : análisis forense recibiendo un WAV por multipart.
- ``/ws/detect``        : análisis en tiempo real (streaming) vía WebSocket.
- ``/review``           : revisión y clasificación de las grabaciones persistidas.
- ``/review/train``     : reentrenamiento del modelo con audios clasificados.

El servidor se ejecuta con uvicorn:
    uv run uvicorn backend.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importación de los cuatro routers que conforman la API.
from backend.api.routes.detection import router as detection_router
from backend.api.routes.review import router as review_router
from backend.api.routes.training import router as training_router
from backend.api.routes.websocket import router as websocket_router

# Instancia principal de la aplicación FastAPI.
app = FastAPI(
    title="Altur Solution API",
    version="0.1.0",
)

# Middleware CORS: permite que el frontend (Vite dev / producción) consuma
# la API sin restricciones de origen cruzado.
# Nota: "*" es aceptable para desarrollo/hackathon; restringir en producción.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de los routers en la aplicación.
# Cada router agrupa un conjunto lógico de endpoints.
app.include_router(detection_router)   # POST /detect, POST /detect_wav
app.include_router(review_router)      # GET/PATCH/DELETE /review/...
app.include_router(training_router)    # POST /review/train, GET /review/train/status
app.include_router(websocket_router)   # WS /ws/detect


@app.get("/")
async def root():
    """Health check básico de la API. Retorna un JSON con un mensaje de estado."""
    return {"message": "Altur Solution API"}