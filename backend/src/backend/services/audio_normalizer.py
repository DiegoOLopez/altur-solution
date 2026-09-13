"""
Normalización de audio para el flujo de streaming (WebSocket /ws/detect).

El frontend captura audio del micrófono del navegador a 8 kHz o 16 kHz
(dependiendo del dispositivo). Este normalizador convierte los chunks
de PCM 16-bit mono a la frecuencia objetivo de 16 kHz que espera el
detector Wav2Vec2:

- 16 kHz → los samples pasan sin modificar (no-op).
- 8 kHz  → se hace resampling en streaming a 16 kHz usando ``soxr``.

El resampling se hace de forma incremental (chunk por chunk) usando
``soxr.ResampleStream``, lo que permite procesar audio en tiempo real
sin necesidad de tener la grabación completa.
"""
import numpy as np
import soxr


# Frecuencia de muestreo objetivo (Wav2Vec2 espera 16 kHz).
TARGET_SAMPLE_RATE = 16_000


class AudioNormalizer:
    """
    Normalizador de audio PCM 16-bit mono a 16 kHz.

    Uso:
        normalizer = AudioNormalizer(sample_rate=8000)
        chunk_16k = normalizer.process(chunk_8k_bytes)

    Atributos:
        sample_rate  : frecuencia de muestreo del audio de entrada.
        _resampler   : instancia de ``soxr.ResampleStream`` (solo si 8 kHz).
    """

    def __init__(self, sample_rate: int):
        """
        Inicializa el normalizador.

        Args:
            sample_rate: Frecuencia de muestreo del audio de entrada.
                         Solo se soportan 8000 Hz y 16000 Hz.

        Raises:
            ValueError: si sample_rate no es 8000 ni 16000.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Unsupported sample rate: {sample_rate} Hz. "
                "Only 8000 Hz and 16000 Hz are supported."
            )

        self.sample_rate = sample_rate
        self._resampler = None

        # Solo crear resampler si el audio es 8 kHz.
        # El resampler de soxr usa calidad "HQ" (alta calidad).
        if sample_rate == 8000:
            self._resampler = soxr.ResampleStream(
                8000,
                TARGET_SAMPLE_RATE,
                1,             # Número de canales (mono).
                dtype="int16",
                quality="HQ",
            )

    def process(self, audio_bytes: bytes) -> bytes:
        """
        Normaliza un chunk de audio a 16 kHz.

        Si el audio ya está en 16 kHz, se devuelve sin modificar.
        Si está en 8 kHz, se resamplea a 16 kHz usando soxr.

        Args:
            audio_bytes: Bytes PCM 16-bit mono.

        Returns:
            Bytes PCM 16-bit mono a 16 kHz.

        Raises:
            ValueError: si el chunk tiene un número impar de bytes
                        (no alineado a muestras de 16 bits).
        """
        if not audio_bytes:
            return b""

        # Validar que el chunk tiene un número par de bytes
        # (cada muestra PCM 16-bit ocupa 2 bytes).
        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                "PCM chunk must contain a whole number of 16-bit samples."
            )

        # Interpretar los bytes como un array de int16.
        samples = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        )

        # Si ya está en 16 kHz, no-op.
        if self.sample_rate == TARGET_SAMPLE_RATE:
            return audio_bytes

        # Resamplear de 8 kHz a 16 kHz con soxr.
        resampled = self._resampler.resample_chunk(samples)

        return resampled.astype(np.int16).tobytes()