"""
Rutas de detección: POST /detect y POST /detect_wav.

Estos endpoints reciben una grabación telefónica y devuelven el
veredicto del detector de voz sintética (humana vs. sintética),
junto con la confianza, el score acumulado y las evidencias
acústicas y comportamentales.

Contratos de entrada:
    - ``POST /detect``     : JSON con ``audio_base64`` (estéreo 8 kHz).
    - ``POST /detect_wav`` : multipart con un archivo WAV.

Contrato de salida (ambos):
    DetectionResponse con is_synthetic, confidence, score_total, etc.
"""
import base64
import binascii

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ...schemas.detection import DetectionResponse
from ...services.detector import AudioDetector


router = APIRouter()


class DetectionJsonRequest(BaseModel):
    """
    Cuerpo de la petición JSON para POST /detect.

    Atributos:
        call_id:      Identificador único de la llamada.
        audio_base64: Audio WAV codificado en Base64.
        sample_rate:  Frecuencia de muestreo (debe ser 8000 Hz).
        channels:     Número de canales (debe ser 2 para estéreo).
    """
    call_id: str
    audio_base64: str
    sample_rate: int
    channels: int


# ============================================================
# Instancia del detector (singleton a nivel de módulo)
# ============================================================

# Se crea una sola instancia del AudioDetector al importar el módulo.
# Esto carga el modelo Wav2Vec2 y el clasificador una sola vez en memoria,
# evitando la sobrecarga de cargarlos en cada petición.
detector = AudioDetector()


def run_detection(audio_bytes: bytes) -> DetectionResponse:
    """
    Ejecuta el detector offline sobre los bytes de audio crudos.

    Args:
        audio_bytes: Contenido binario del archivo WAV.

    Returns:
        DetectionResponse con el veredicto del modelo.

    Raises:
        HTTPException 400: si el payload está vacío o el audio es inválido.
        HTTPException 500: si ocurre un error interno en el detector.
    """
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
    """
    Endpoint oficial JSON/Base64 usado por el juez externo del hackathon.

    Valida que el audio sea estéreo a 8000 Hz, decodifica el Base64
    y ejecuta la detección offline.

    Args:
        payload: DetectionJsonRequest con call_id, audio_base64, sample_rate, channels.

    Returns:
        DetectionResponse con el veredicto del modelo.
    """
    # Validar parámetros de formato esperados por la norma del reto.
    if payload.sample_rate != 8000 or payload.channels != 2:
        raise HTTPException(
            status_code=400,
            detail="Expected stereo audio at 8000 Hz.",
        )

    # Decodificar el audio Base64 a bytes crudos.
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
    """
    Endpoint multipart WAV usado por el frontend (flujo de análisis forense).

    Acepta un archivo WAV subido como multipart/form-data bajo el campo
    ``file``. Valida el nombre del archivo y ejecuta la detección offline.

    Args:
        file: Archivo WAV subido por el usuario.

    Returns:
        DetectionResponse con el veredicto del modelo.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    return run_detection(await file.read())