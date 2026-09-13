import base64
import binascii

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ...schemas.detection import DetectionResponse
from ...services.detector import AudioDetector


router = APIRouter()


class DetectionJsonRequest(BaseModel):
    call_id: str
    audio_base64: str
    sample_rate: int
    channels: int


# ============================================================
# Detector
# ============================================================

detector = AudioDetector()


def run_detection(audio_bytes: bytes) -> DetectionResponse:
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The audio payload is empty.",
        )

    # Ejecutar el detector offline y traducir los errores a respuestas HTTP.
    try:
        result = detector.detect_offline(audio_bytes)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(exc)}",
        )

    return result


@router.post("/detect", response_model=DetectionResponse)
async def detect_audio(payload: DetectionJsonRequest):
    """Official JSON/Base64 contract used by the external judge."""
    if payload.sample_rate != 8000 or payload.channels != 2:
        raise HTTPException(
            status_code=400,
            detail="Expected stereo audio at 8000 Hz.",
        )

    try:
        audio_bytes = base64.b64decode(payload.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="audio_base64 is not valid Base64.",
        ) from exc

    return run_detection(audio_bytes)


@router.post("/detect_wav", response_model=DetectionResponse)
async def detect_wav(file: UploadFile = File(...)):
    """Multipart WAV contract used by the frontend Audio Forensics flow."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    return run_detection(await file.read())