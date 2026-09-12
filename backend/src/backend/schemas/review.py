from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.models.audio import AudioStatus


class AudioReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ref: str
    title: str
    duration: float
    sample_rate: int
    channels: int
    is_synthetic: bool
    confidence: float | None
    score_total: float | None
    status: AudioStatus
    created_at: datetime
    audio_url: str


class AudioClassificationRequest(BaseModel):
    classification: Literal["synthetic", "real"]


class AudioMutationResponse(BaseModel):
    message: str
    audio: AudioReviewResponse