import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

TARGET_SAMPLE_RATE = 16000

MIN_SEGMENT_DURATION = 0.5
MAX_SEGMENT_DURATION = 10.0
MERGE_GAP = 0.5

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = load_silero_vad()

    return _model


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float]]:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, "
            f"received {sample_rate} Hz."
        )

    if audio.dtype != np.float32:
        raise ValueError(
            f"Expected float32 audio, received {audio.dtype}."
        )

    if audio.size == 0:
        return []

    model = _get_model()

    audio_tensor = torch.from_numpy(audio)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
    )

    segments = [
        (
            timestamp["start"] / sample_rate,
            timestamp["end"] / sample_rate,
        )
        for timestamp in speech_timestamps
    ]

    return merge_speech_segments(segments)


def merge_speech_segments(
    segments: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not segments:
        return []

    merged = []

    current_start, current_end = segments[0]

    for start, end in segments[1:]:
        gap = start - current_end

        if gap <= MERGE_GAP:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start = start
            current_end = end

    merged.append((current_start, current_end))

    filtered = [
        (start, end)
        for start, end in merged
        if end - start >= MIN_SEGMENT_DURATION
    ]

    final_segments = []

    for start, end in filtered:
        duration = end - start

        if duration <= MAX_SEGMENT_DURATION:
            final_segments.append((start, end))
            continue

        current = start

        while current < end:
            segment_end = min(current + MAX_SEGMENT_DURATION, end)

            if segment_end - current >= MIN_SEGMENT_DURATION:
                final_segments.append((current, segment_end))

            current = segment_end

    return final_segments