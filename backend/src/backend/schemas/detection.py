from pydantic import BaseModel


class DetectionResponse(BaseModel):
    """
    Resultado de la detección de voz sintética.
    """

    # Resultado principal del detector
    is_synthetic: bool

    # Confianza de la clasificación
    confidence: float

    # Score acumulado del modelo
    score_total: float

    # Evidencia acústica acumulada
    llr_acoustic_cum: float

    # Evidencia comportamental acumulada
    llr_behavioral_cum: float

    # Cantidad de segmentos acústicos analizados
    n_acoustic_segments: int

    # Cantidad de eventos comportamentales analizados
    n_behavioral_events: int

    # Umbral utilizado para tomar la decisión
    eta: float