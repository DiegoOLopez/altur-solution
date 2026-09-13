"""
Rutas de revisión /review.

Permiten a un humano (o al flujo de segundo filtro) consultar las
grabaciones persistidas, reproducirlas y clasificarlas. Cada grabación
tiene un estado (pendiente, revisado o eliminado) que alimenta tanto el
Review Hub del frontend como el dataset de entrenamiento del modelo.

Endpoints:
    GET    /review/db/health                         — Health check de la BD.
    GET    /review/audios                            — Lista de grabaciones.
    GET    /review/stats                             — Conteos de pendientes/revisados.
    GET    /review/audios/{audio_id}/stream           — Streaming del audio desde MinIO.
    PATCH  /review/audios/{audio_id}/classification   — Clasificar audio como sintético/real.
    DELETE /review/audios/{audio_id}                  — Eliminación lógica.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from minio import Minio
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
    Convierte un registro ORM ``Audio`` en la representación Pydantic
    que consume el frontend, incluyendo la URL relativa para reproducir
    la grabación vía streaming desde MinIO.

    Args:
        audio: Instancia del modelo SQLAlchemy Audio.

    Returns:
        AudioReviewResponse listo para serializar como JSON.
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

    Returns:
        JSON con status "ok" si la BD responde.

    Raises:
        HTTPException 503 si la conexión falla.
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

    Filtros opcionales:
        status:          Filtrar por estado específico (revisado, no_revisado, deleted).
        include_deleted: Si es True, incluye audios eliminados (por defecto False).

    Returns:
        Lista de AudioReviewResponse.
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

    Returns:
        JSON con pending, reviewed y total.
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
    Busca una grabación por su ID. Retorna HTTP 404 si no existe
    o si fue marcada como eliminada.

    Args:
        audio_id: ID numérico de la grabación.
        db:       Sesión de base de datos.

    Returns:
        Instancia del modelo Audio.

    Raises:
        HTTPException 404 si el audio no existe o fue eliminado.
    """
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if audio is None or audio.status == AudioStatus.DELETED:
        raise HTTPException(status_code=404, detail="Audio not found.")
    return audio


# ============================================================
# Cliente MinIO para streaming de audio
# ============================================================

# Configuración del cliente MinIO.
# Las credenciales y el endpoint corresponden al entorno de desarrollo
# definido en docker-compose.yml.
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False
)

# Nombre del bucket donde se almacenan las grabaciones.
BUCKET_NAME = "grabaciones"


@router.get("/audios/{audio_id}/stream")
def stream_audio(audio_id: int, db: Session = Depends(get_db)):
    """
    Sirve la grabación guardada en MinIO como respuesta de streaming.

    Descarga el archivo WAV del bucket de MinIO usando la clave
    ``storage_key`` del registro en la BD y lo envía en chunks al
    cliente para evitar cargar todo el archivo en memoria.

    Args:
        audio_id: ID numérico de la grabación.
        db:       Sesión de base de datos.

    Returns:
        StreamingResponse con content-type audio/wav.

    Raises:
        HTTPException 404 si el audio no existe en MinIO.
    """
    audio = get_audio_or_404(audio_id, db)

    try:
        # Obtener el objeto (audio) desde MinIO usando el storage_key.
        response = minio_client.get_object(BUCKET_NAME, audio.storage_key)

        # Función generadora para leer en chunks de 32 KB
        # y hacer streaming eficiente sin cargar todo en RAM.
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

    Este endpoint alimenta el dataset de entrenamiento: cada audio
    clasificado por un humano puede usarse posteriormente para
    reentrenar el modelo vía POST /review/train.

    Args:
        audio_id: ID numérico de la grabación.
        payload:  AudioClassificationRequest con el valor "synthetic" o "real".
        db:       Sesión de base de datos.

    Returns:
        AudioMutationResponse con mensaje de éxito y el audio actualizado.
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
    Eliminación lógica: marca la grabación como borrada sin removerla
    del bucket de MinIO ni del disco.

    Args:
        audio_id: ID numérico de la grabación.
        db:       Sesión de base de datos.

    Returns:
        AudioMutationResponse con mensaje de confirmación.
    """
    audio = get_audio_or_404(audio_id, db)
    audio.status = AudioStatus.DELETED
    db.commit()
    db.refresh(audio)

    return {
        "message": "Audio marked as deleted.",
        "audio": serialize_audio(audio),
    }