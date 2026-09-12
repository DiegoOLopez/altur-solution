from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas.detection import DetectionResponse
from backend.services.audio import process_wav

router = APIRouter(
    prefix="/detect",
    tags=["Detection"],
)


@router.post("", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Recibe un archivo WAV y lo normaliza a 16 kHz.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required.",
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported.",
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        (
            original_sample_rate,
            final_sample_rate,
            converted,
            _processed_audio,
        ) = process_wav(audio_bytes)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return DetectionResponse(
        message=(
            "Audio converted from 8000 Hz to 16000 Hz."
            if converted
            else "Audio was already 16000 Hz. No conversion was necessary."
        ),
        original_sample_rate=original_sample_rate,
        final_sample_rate=final_sample_rate,
        converted=converted,
    )