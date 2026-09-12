from pathlib import Path
import sys

import joblib
import numpy as np


DETECTOR_ROOT = (
    Path(__file__).resolve().parents[4] / "detector"
)

if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_models"
    / "model.pkl"
)


class AudioDetector:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.session = self.model.new_session()

    def process_audio(
        self,
        audio: bytes,
        sample_rate: int,
    ) -> dict | None:
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

        # Obtenemos el snapshot calibrado del detector.
        return self.session.current_snapshot()