"""
Detector de voz sintética de AuraVoice.

Es el corazón del sistema: carga el modelo entrenado ``ai_models/model.pkl``
una sola vez al arrancar el servicio y expone dos formas de análisis:

- ``detect_offline`` (lote): analiza una llamada completa. Se usa en
  ``POST /detect`` y recibe el WAV estéreo original (Canal 0 = llamante,
  Canal 1 = agente) tal cual llega del frontend. El modelo se encarga
  internamente de cargar el WAV, resamplear a 16 kHz, separar los canales,
  extraer las características acústicas y comportamentales, calcular el LLR,
  calibrar y devolver el veredicto final. Es importante NO extraer solo el
  Canal 0 antes de llamar al modelo: el análisis comportamental necesita
  los dos canales para observar turnos e interrupciones.

- ``process_audio`` (streaming): analiza chunks en tiempo real. Se usa en el
  WebSocket ``/ws/detect`` y recibe PCM mono de 16 kHz. Cada chunk alimenta
  una sesión de streaming que acumula la evidencia y devuelve el snapshot
  parcial con el estado actual de la llamada.
"""
from pathlib import Path
import sys

import joblib
import numpy as np


# ------------------------------------------------------------------
# Ubicación del detector externo
# ------------------------------------------------------------------
# El modelo se encuentra en el paquete "detector" de este repositorio;
# se agrega su ruta al sys.path para poder importarlo sin instalarlo.
DETECTOR_ROOT = Path(__file__).resolve().parents[4] / "detector"

if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))


# ------------------------------------------------------------------
# Modelo entrenado
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_models"
    / "model_1_2.pkl"
)


class AudioDetector:
    """
    Detector de voz sintética.

    Ambos modos (streaming y lote) usan el MISMO modelo cargado en memoria;
    la única diferencia es cómo se alimenta: por chunks PCM en el modo
    streaming o con un WAV completo en el modo offline.
    """

    def __init__(self):
        """
        Carga el modelo entrenado una sola vez al iniciar el servicio.
        """
        self.model = joblib.load(MODEL_PATH)

        # Sesión de streaming usada por el WebSocket /ws/detect.
        self.session = self.model.new_session()

    def process_audio(
        self,
        audio: bytes,
        sample_rate: int,
    ) -> dict | None:
        """
        Procesa un chunk de audio en modo streaming.

        Recibe PCM mono de 16 kHz y alimenta la sesión de streaming del
        modelo. Devuelve el snapshot acumulado cuando hay suficiente
        evidencia acumulada para emitir una inferencia, o ``None`` en caso
        contrario.

        Este método se mantiene para el WebSocket.
        """

        if not audio:
            return None

        if sample_rate != 16_000:
            raise ValueError(
                "AudioDetector expects 16000 Hz audio."
            )

        # PCM 16-bit mono -> flotantes normalizados en [-1, 1].
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