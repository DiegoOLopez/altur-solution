import io
import wave

import numpy as np


def extract_channel0(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, received {sample_rate} Hz."
        )

    if sample_width != 2:
        raise ValueError(
            "Only 16-bit PCM WAV files are supported."
        )

    audio = np.frombuffer(frames, dtype=np.int16)

    if channels == 1:
        channel0 = audio

    elif channels == 2:
        audio = audio.reshape(-1, 2)
        channel0 = audio[:, 0]

    else:
        raise ValueError(
            f"Unsupported number of channels: {channels}."
        )

    channel0 = channel0.astype(np.float32) / 32768.0

    return channel0, sample_rate