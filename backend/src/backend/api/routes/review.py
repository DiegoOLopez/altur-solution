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
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if audio is None or audio.status == AudioStatus.DELETED:
        raise HTTPException(status_code=404, detail="Audio not found.")
    return audio


@router.get("/audios/{audio_id}/stream")
def stream_audio(audio_id: int, db: Session = Depends(get_db)):
    audio = get_audio_or_404(audio_id, db)
    storage_root = Path(settings.audio_storage_root).resolve()
    audio_path = (storage_root / audio.storage_key).resolve()

    if storage_root not in audio_path.parents:
        raise HTTPException(status_code=400, detail="Invalid audio storage key.")
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=audio_path.name,
    )


@router.patch(
    "/audios/{audio_id}/classification",
    response_model=AudioMutationResponse,
)
def classify_audio(
    audio_id: int,
    payload: AudioClassificationRequest,
    db: Session = Depends(get_db),
):
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
    audio = get_audio_or_404(audio_id, db)
    audio.status = AudioStatus.DELETED
    db.commit()
    db.refresh(audio)

    return {
        "message": "Audio marked as deleted.",
        "audio": serialize_audio(audio),
    }