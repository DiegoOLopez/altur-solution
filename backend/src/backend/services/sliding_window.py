class SlidingWindow:
    """
    Mantiene una ventana deslizante de audio PCM.

    Configuración:
        Sample rate: 16 kHz
        Mono
        16-bit PCM
        Window: 1 segundo
        Hop: 100 ms
    """

    SAMPLE_RATE = 16_000
    SAMPLE_WIDTH = 2
    CHANNELS = 1

    WINDOW_SECONDS = 1.0
    HOP_SECONDS = 0.1

    WINDOW_SIZE = int(
        SAMPLE_RATE
        * SAMPLE_WIDTH
        * CHANNELS
        * WINDOW_SECONDS
    )

    HOP_SIZE = int(
        SAMPLE_RATE
        * SAMPLE_WIDTH
        * CHANNELS
        * HOP_SECONDS
    )

    def __init__(self):
        self.buffer = bytearray()
        self.window_number = 0

    def add(self, audio_bytes: bytes) -> list[bytes]:
        """
        Agrega audio al buffer.

        Returns:
            Lista de ventanas completas listas para inferencia.
        """

        self.buffer.extend(audio_bytes)

        windows = []

        while len(self.buffer) >= self.WINDOW_SIZE:
            self.window_number += 1

            window = bytes(
                self.buffer[:self.WINDOW_SIZE]
            )

            windows.append(window)

            # Conservamos 900 ms y avanzamos 100 ms.
            del self.buffer[:self.HOP_SIZE]

        return windows