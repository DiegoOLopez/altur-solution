"""
Schemas Pydantic del Review Hub.

Definen los modelos de request/response para los endpoints de revisión
y clasificación de grabaciones.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class AudioReviewResponse(BaseModel):
    """
    Representación de un audio para el Review Hub del frontend.

    Incluye los metadatos de la grabación, el resultado de la detección
    y la URL de streaming para reproducirla en el navegador.

    Atributos:
        id           : PK numérico.
        ref          : UUID corto de la grabación.
        title        : Nombre amigable.
        duration     : Duración en segundos.
        sample_rate  : Frecuencia de muestreo.
        channels     : Número de canales.
        is_synthetic : True si fue clasificada como sintética.
        confidence   : Confianza del detector.
        score_total  : Score acumulado del detector.
        status       : Estado actual (no_revisado, revisado, deleted).
        created_at   : Timestamp de creación.
        audio_url    : URL relativa para streaming (/review/audios/{id}/stream).
    """
    id: int
    ref: str
    title: str
    duration: float
    sample_rate: int
    channels: int
    is_synthetic: bool
    confidence: float | None
    score_total: float | None
    status: str
    created_at: datetime | None
    audio_url: str


class AudioClassificationRequest(BaseModel):
    """
    Petición para clasificar un audio manualmente.

    Atributos:
        classification: "synthetic" si el audio es voz sintética,
                        "real" si es voz humana legítima.
    """
    classification: Literal["synthetic", "real"]


class AudioMutationResponse(BaseModel):
    """
    Respuesta genérica para mutaciones (clasificar, eliminar).

    Atributos:
        message : Mensaje de confirmación.
        audio   : Estado actualizado del audio.
    """
    message: str
    audio: AudioReviewResponse