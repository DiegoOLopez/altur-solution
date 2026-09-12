from fastapi import APIRouter, File, HTTPException, UploadFile

from ...services.detector import AudioDetector
from ...schemas.detection import DetectionResponse


router = APIRouter()


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
    file: UploadFile = File(...),
):
    """
    Detecta si una llamada contiene una voz sintética.

    El modelo recibe el WAV estéreo original para poder
    analizar tanto las características acústicas del caller
    como el comportamiento conversacional entre caller y agente.
    """

    # --------------------------------------------------------
    # Validar extensión
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported.",
        )


    # --------------------------------------------------------
    # Leer archivo
    # --------------------------------------------------------

    audio_bytes = await file.read()

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