"""
Schemas Pydantic de detección.

Define el modelo de respuesta ``DetectionResponse`` que retornan
los endpoints POST /detect, POST /detect_wav y el WebSocket /ws/detect.
"""
from pydantic import BaseModel


class DetectionResponse(BaseModel):
    """
    Respuesta del detector de voz sintética.

    Atributos:
        is_synthetic          : True si la voz fue clasificada como sintética.
        confidence            : Nivel de confianza del detector (0.0 a 1.0).
        score_total           : Score total acumulado (LLR acústico + comportamental).
        llr_acoustic_cum      : Log-likelihood ratio acústico acumulado.
        llr_behavioral_cum    : Log-likelihood ratio comportamental acumulado.
        n_acoustic_segments   : Cantidad de segmentos acústicos analizados.
        n_behavioral_events   : Cantidad de eventos comportamentales detectados.
        eta                   : Umbral de decisión del clasificador.
        message               : Mensaje descriptivo del resultado.
    """
    is_synthetic: bool
    confidence: float
    score_total: float
    llr_acoustic_cum: float
    llr_behavioral_cum: float
    n_acoustic_segments: int
    n_behavioral_events: int
    eta: float
    message: str