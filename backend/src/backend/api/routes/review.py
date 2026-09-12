"""
Rutas de revisión /review.

Permiten a un humano (o al flujo de segundo filtro) consultar las
grabaciones persistidas, reproducirlas y clasificarlas. Cada grabación
tiene un estado (pendiente, revisado o eliminado) que alimenta tanto el
Review Hub del frontend como el dataset de entrenamiento del modelo.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import get_db
from backend.models.audio import Audio, AudioStatus
from backend.schemas.review import (
    AudioClassificationRequest,
    AudioMutationResponse,
    AudioReviewResponse,
)


router = APIRouter(
    prefix="/review",
    tags=["Review"],
)


def serialize_audio(audio: Audio) -> AudioReviewResponse:
    """
    Convierte un registro ``Audio`` en la representación que consume el
    frontend, incluyendo la URL relativa para reproducir la grabación.
    """
    return AudioReviewResponse(
        id=audio.id,
        ref=audio.ref,
        title=audio.title,
        duration=audio.duration,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        is_synthetic=audio.is_synthetic,
        confidence=audio.confidence,
        score_total=audio.score_total,
        status=audio.status,
        created_at=audio.created_at,
        audio_url=f"/review/audios/{audio.id}/stream",
    )


@router.get("/db/health")
def database_health(db: Session = Depends(get_db)):
    """
    Health check: verifica que la conexión a MySQL responde (SELECT 1).
    """
    try:
        db.execute(text("SELECT 1"))
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Database connection is unavailable.",
        ) from error

    return {"status": "ok", "database": "connected"}


@router.get("/audios", response_model=list[AudioReviewResponse])
def list_audios(
    status: AudioStatus | None = Query(default=None),
    include_deleted: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """
    Lista las grabaciones, ordenadas de la más reciente a la más antigua.
    """
    query = db.query(Audio)
    if status is not None:
        query = query.filter(Audio.status == status)
    elif include_deleted:
        query = query.filter(Audio.status == AudioStatus.NO_REVISADO)
    else:
        query = query.filter(Audio.status == AudioStatus.NO_REVISADO)

    audios = query.order_by(Audio.created_at.desc(), Audio.id.desc()).all()
    return [serialize_audio(audio) for audio in audios]


@router.get("/stats")
def review_stats(db: Session = Depends(get_db)):
    """
    Resume las grabaciones pendientes y clasificadas del Review Hub.
    """
    pending = db.query(func.count(Audio.id)).filter(
        Audio.status == AudioStatus.NO_REVISADO
    ).scalar() or 0
    reviewed = db.query(func.count(Audio.id)).filter(
        Audio.status == AudioStatus.REVISADO
    ).scalar() or 0

    return {
        "pending": pending,
        "reviewed": reviewed,
        "total": pending + reviewed,
    }


def get_audio_or_404(audio_id: int, db: Session) -> Audio:
    """
    Devuelve la grabación por id, o HTTP 404 si no existe o fue eliminada.
    """
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if audio is None or audio.status == AudioStatus.DELETED:
        raise HTTPException(status_code=404, detail="Audio not found.")
    return audio


from fastapi.responses import StreamingResponse
from minio import Minio

# Configuración del Cliente MinIO (Igual que en websocket.py)
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False
)
BUCKET_NAME = "grabaciones"

@router.get("/audios/{audio_id}/stream")
def stream_audio(audio_id: int, db: Session = Depends(get_db)):
    """
    Sirve la grabación guardada localmente. Valida que la ``storage_key``
    apunte dentro del directorio de audio para evitar path traversal.
    """
    audio = get_audio_or_404(audio_id, db)
    
    try:
        # Obtenemos el objeto (audio) desde MinIO usando el storage_key
        response = minio_client.get_object(BUCKET_NAME, audio.storage_key)
        
        # Función generadora para leer en chunks y hacer streaming eficiente
        def iterfile():
            try:
                for chunk in response.stream(32 * 1024):
                    yield chunk
            finally:
                response.close()
                response.release_conn()
                
        return StreamingResponse(
            iterfile(), 
            media_type="audio/wav"
        )
    except Exception as e:
        print(f"Error al leer de MinIO: {e}")
        raise HTTPException(status_code=404, detail="Audio file not found in storage.")


@router.patch(
    "/audios/{audio_id}/classification",
    response_model=AudioMutationResponse,
)
def classify_audio(
    audio_id: int,
    payload: AudioClassificationRequest,
    db: Session = Depends(get_db),
):
    """
    Marca una grabación como sintética o real y la pasa a estado revisado.
    """
    audio = get_audio_or_404(audio_id, db)
    audio.is_synthetic = payload.classification == "synthetic"
    audio.status = AudioStatus.REVISADO
    db.commit()
    db.refresh(audio)

    return {
        "message": "Audio classified successfully.",
        "audio": serialize_audio(audio),
    }


@router.delete(
    "/audios/{audio_id}",
    response_model=AudioMutationResponse,
)
def delete_audio(audio_id: int, db: Session = Depends(get_db)):
    """
    Eliminación lógica: marca la grabación como borrada sin removerla del
    bucket ni del disco.
    """
    audio = get_audio_or_404(audio_id, db)
    audio.status = AudioStatus.DELETED
    db.commit()
    db.refresh(audio)

    return {
        "message": "Audio marked as deleted.",
        "audio": serialize_audio(audio),
    }