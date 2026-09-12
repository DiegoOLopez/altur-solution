from pydantic import BaseModel


class ModelSegment(BaseModel):
    segment: int
    start: float
    end: float
    duration: float
    feature_vector: list[float]


class CallEmbedding(BaseModel):
    mean: list[float]
    std: list[float]
    mean_norm: float
    std_norm: float


class ModelFeatures(BaseModel):
    sample_rate: int
    feature_names: list[str]
    feature_matrix: list[list[float]]
    segments: list[ModelSegment]
    call_embedding: CallEmbedding