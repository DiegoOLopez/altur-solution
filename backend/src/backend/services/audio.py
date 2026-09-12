import io
import wave

import numpy as np
from scipy.signal import resample_poly


def process_wav(audio_bytes: bytes) -> tuple[int, int, bool, bytes]:
    """
    Procesa un archivo WAV.

    - 8000 Hz -> convierte a 16000 Hz
    - 16000 Hz -> no realiza cambios

    Returns:
        original_sample_rate
        final_sample_rate
        converted
        processed_audio_bytes
    """

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_rate not in (8000, 16000):
        raise ValueError(
            f"Unsupported sample rate: {sample_rate} Hz. "
            "Only 8000 Hz and 16000 Hz are supported."
        )

    # Por ahora solo trabajamos con PCM de 16 bits.
    if sample_width != 2:
        raise ValueError(
            "Only 16-bit PCM WAV files are supported."
        )

    audio = np.frombuffer(frames, dtype=np.int16)

    # Reorganizamos el audio si tiene más de un canal.
    if channels > 1:
        audio = audio.reshape(-1, channels)

    if sample_rate == 16000:
        return (
            16000,
            16000,
            False,
            audio_bytes,
        )

    # 8000 Hz -> 16000 Hz
    resampled = resample_poly(audio, 2, 1)

    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(16000)
        wav_file.writeframes(resampled.tobytes())

    return (
        8000,
        16000,
        True,
        output.getvalue(),
    )