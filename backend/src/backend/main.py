from fastapi import FastAPI

from backend.api.routes.detection import router as detection_router
from backend.api.routes.websocket import router as websocket_router

app = FastAPI(
    title="Altur Solution API",
    version="0.1.0",
)

app.include_router(detection_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    return {"message": "Altur Solution API"}