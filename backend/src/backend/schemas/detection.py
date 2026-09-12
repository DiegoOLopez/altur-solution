from pydantic import BaseModel


class DetectionResponse(BaseModel):
    message: str
    original_sample_rate: int
    final_sample_rate: int
    converted: bool