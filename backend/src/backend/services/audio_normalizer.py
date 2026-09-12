# Este servicio es exclusivo del streaming

import numpy as np
import soxr


TARGET_SAMPLE_RATE = 16_000


class AudioNormalizer:
    """
    Normaliza audio PCM 16-bit mono a 16 kHz.

    - 16 kHz -> pasa los samples sin modificar.
    - 8 kHz  -> hace resampling streaming a 16 kHz.
    """

    def __init__(self, sample_rate: int):
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Unsupported sample rate: {sample_rate} Hz. "
                "Only 8000 Hz and 16000 Hz are supported."
            )

        self.sample_rate = sample_rate

        self._resampler = None

        if sample_rate == 8000:
            self._resampler = soxr.ResampleStream(
                8000,
                TARGET_SAMPLE_RATE,
                1,
                dtype="int16",
                quality="HQ",
            )

    def process(self, audio_bytes: bytes) -> bytes:
        """
        Recibe un chunk de PCM 16-bit mono y devuelve PCM 16 kHz.
        """

        if not audio_bytes:
            return b""

        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                "PCM chunk must contain a whole number of 16-bit samples."
            )

        samples = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        )

        # Ya está en 16 kHz.
        if self.sample_rate == TARGET_SAMPLE_RATE:
            return audio_bytes

        # 8 kHz -> 16 kHz
        resampled = self._resampler.resample_chunk(samples)

        return resampled.astype(np.int16).tobytes()