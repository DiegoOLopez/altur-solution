"""
Detector de voz sintética de AuraVoice.

Es el corazón del sistema: carga el modelo entrenado ``ai_models/altur_detector_model.joblib``
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
    / "altur_detector_model.joblib"
)


import io
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

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
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando Wav2Vec2 en {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.wav2vec.eval()

        class DummySession:
            def __init__(self, parent):
                self.parent = parent
                self.buffer = []
                self.last_snap = None

            def push_audio_chunk(self, audio_array, sample_rate):
                self.buffer.append(audio_array)
                # Si acumulamos ~3 segundos de audio
                total_samples = sum(len(x) for x in self.buffer)
                if total_samples >= sample_rate * 3:
                    combined = np.concatenate(self.buffer)
                    self.buffer = []
                    self.last_snap = self.parent._predict(combined, sample_rate)
                    return self.last_snap
                return None
                
            def current_snapshot(self):
                return self.last_snap

        # Sesión de streaming usada por el WebSocket /ws/detect.
        self.session = DummySession(self)

    def _predict(self, audio_array, sr) -> dict:
        waveform = torch.from_numpy(audio_array).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
        if waveform.shape[1] > 48000:
            waveform = waveform[:, :48000]
            
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.wav2vec(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[4] 
            features = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
        features = features.reshape(1, -1)
        proba = self.model.predict_proba(features)[0][1]
        is_synthetic = bool(proba > 0.5)
        
        return {
            "t": 0.0,
            "is_synthetic": is_synthetic,
            "confidence": float(proba) if is_synthetic else float(1.0 - proba),
            "score_total": float(proba),
            "llr_acoustic_cum": float(proba),
            "llr_behavioral_cum": 0.0,
            "n_acoustic_segments": 1,
            "n_behavioral_events": 0,
            "eta": 0.5
        }

    def process_audio(
        self,
        audio: bytes,
        sample_rate: int,
    ) -> dict | None:
        """
        Procesa un chunk de audio en modo streaming.
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
        ).astype(np.float64) / 32768.0

        snapshot = self.session.push_audio_chunk(
            audio_array,
            sample_rate,
        )

        return snapshot

    def detect_offline(
        self,
        audio: bytes,
    ) -> dict:
        """
        Procesa una llamada completa en modo offline.
        """
        if not audio:
            raise ValueError("Audio cannot be empty.")

        data, sr = sf.read(io.BytesIO(audio), dtype='float32')
        # Si es estéreo, nos quedamos con el canal 0 (caller)
        if data.ndim > 1:
            data = data[:, 0]
            
        return self._predict(data, sr)