"""
Endpoint POST /detect: análisis forense de una llamada completa.

Recibe el WAV estéreo original (Canal 0 = llamante, Canal 1 = agente)
como multipart/form-data bajo el campo ``file`` y devuelve el veredicto
estructurado por ``DetectionResponse``. El audio no se preprocesa en el
API: el modelo recibe el archivo tal cual llega del cliente y es quien
hace internamente el resampleo, la separación de canales y el análisis
acústico y comportamental.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from ...schemas.detection import DetectionResponse
from ...services.detector import AudioDetector


router = APIRouter()

# Detector compartido: el modelo se carga una sola vez al arrancar el servicio.
detector = AudioDetector()


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

    # Validar que el archivo tenga nombre y extensión WAV.
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

    # Leer el contenido del archivo subido.
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
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