import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

_model = None


def _get_model() -> EncoderClassifier:
    global _model

    if _model is None:
        _model = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
        )

    return _model


def extract_voice_embedding(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    if audio.dtype != np.float32:
        raise ValueError("Audio must be float32.")

    if sample_rate != 16000:
        raise ValueError("Sample rate must be 16000 Hz.")

    if audio.size == 0:
        raise ValueError("Audio cannot be empty.")

    model = _get_model()

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    with torch.inference_mode():
        embedding = model.encode_batch(audio_tensor)

    embedding = (
        embedding
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return embedding


def aggregate_voice_embeddings(
    embeddings: list[np.ndarray],
    durations: list[float],
) -> dict[str, np.ndarray | float]:
    if not embeddings:
        raise ValueError("At least one embedding is required.")

    if not durations:
        raise ValueError("At least one duration is required.")

    if len(embeddings) != len(durations):
        raise ValueError(
            "The number of embeddings must match the number of durations."
        )

    embedding_matrix = np.stack(embeddings).astype(np.float32)
    duration_array = np.asarray(durations, dtype=np.float32)

    if np.any(duration_array <= 0):
        raise ValueError("All durations must be greater than zero.")

    weights = duration_array / np.sum(duration_array)

    mean_embedding = np.sum(
        embedding_matrix * weights[:, np.newaxis],
        axis=0,
    )

    std_embedding = np.sqrt(
        np.sum(
            weights[:, np.newaxis]
            * np.square(
                embedding_matrix - mean_embedding
            ),
            axis=0,
        )
    )

    embedding_norms = np.linalg.norm(
        embedding_matrix,
        axis=1,
    )

    mean_norm = float(
        np.sum(embedding_norms * weights)
    )

    std_norm = float(
        np.sqrt(
            np.sum(
                weights
                * np.square(
                    embedding_norms - mean_norm
                )
            )
        )
    )

    return {
        "mean_embedding": mean_embedding,
        "std_embedding": std_embedding,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
    }