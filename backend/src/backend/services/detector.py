from pathlib import Path
import sys

import joblib
import numpy as np


# ============================================================
# Ubicación del detector externo
# ============================================================

DETECTOR_ROOT = (
    Path(__file__).resolve().parents[4] / "detector"
)

if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))


# ============================================================
# Modelo entrenado
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_models"
    / "model_1_2.pkl"
)


class AudioDetector:
    def __init__(self):
        """
        Carga el modelo entrenado una sola vez al iniciar
        el servicio.
        """

        self.model = joblib.load(MODEL_PATH)

        # Sesión utilizada por el detector en modo streaming.
        self.session = self.model.new_session()


    def process_audio(
        self,
        audio: bytes,
        sample_rate: int,
    ) -> dict | None:
        """
        Procesa audio en modo streaming.

        Este método se mantiene para el WebSocket.
        """

        if not audio:
            return None

        if sample_rate != 16_000:
            raise ValueError(
                "AudioDetector expects 16000 Hz audio."
            )

        audio_array = np.frombuffer(
            audio,
            dtype=np.int16,
        ).astype(np.float64)

        snapshot = self.session.push_audio_chunk(
            audio_array,
            sample_rate,
        )

        if snapshot is None:
            return None

        return self.session.current_snapshot()


    def detect_offline(
        self,
        audio: bytes,
    ) -> dict:
        """
        Procesa una llamada completa en modo offline.

        El modelo recibe directamente el WAV original porque
        internamente se encarga de:

        - cargar el WAV
        - convertir/resamplear a 16 kHz
        - separar caller y agente
        - analizar características acústicas
        - analizar comportamiento conversacional
        - calcular LLR
        - aplicar calibración
        - generar el resultado final

        Es importante NO extraer solamente Channel 0 antes
        de llamar al modelo, ya que el modelo necesita el
        audio estéreo para su análisis comportamental.
        """

        if not audio:
            raise ValueError("Audio cannot be empty.")

        result = self.model.predict_offline(audio)

        return result