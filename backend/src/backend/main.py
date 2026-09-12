from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.detection import router as detection_router
from backend.api.routes.websocket import router as websocket_router

app = FastAPI(
    title="Altur Solution API",
    version="0.1.0",
)

# Permitir que el frontend (Vite dev / producción) consuma la API.
# Nota: "*" es aceptable para desarrollo/hackathon; restringir en producción.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    return {"message": "Altur Solution API"}