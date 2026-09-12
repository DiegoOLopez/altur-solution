import base64
import binascii

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ...services.detector import AudioDetector
from ...schemas.detection import DetectionResponse


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


# ============================================================
# POST /detect
# ============================================================

@router.post(
    "/detect",
    response_model=DetectionResponse,
)
async def detect_audio(
    request: Request,
):
    """
    Detecta si una llamada contiene una voz sintética.

    El modelo recibe el WAV estéreo original para poder
    analizar tanto las características acústicas del caller
    como el comportamiento conversacional entre caller y agente.
    """

    content_type = request.headers.get("content-type", "").lower()
    file = None
    payload = None

    if content_type.startswith("application/json"):
        try:
            payload = DetectionJsonRequest.model_validate(await request.json())
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=400, detail="A multipart WAV file is required.")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV files are supported.")

        audio_bytes = await file.read()
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json or multipart/form-data.",
        )

    if payload is not None:
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
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )


    # --------------------------------------------------------
    # Ejecutar detector
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Regresar resultado
    # --------------------------------------------------------

    return result