"""
Pipeline completo: junta extracción de features + densidades + LLR +
calibración en un solo objeto serializable (el "binario" del modelo).

Expone dos formas de uso:
  - `VoiceAuthenticityDetector.predict_offline(...)`: procesa un WAV completo
    de una sola vez (modo batch, usado por el endpoint POST /detect).
  - `StreamingSession`: procesa audio y eventos de turno incrementalmente,
    tal como llegarían en una llamada real, para alimentar el dashboard en
    vivo y simular el comportamiento en tiempo real exigido por el reto.

En ambos casos el cómputo interno es *el mismo*: el modo offline simplemente
alimenta una StreamingSession de principio a fin sin dormir entre pasos.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field

import numpy as np
import librosa
import soundfile as sf
import joblib

from features.acoustic import segment_features, is_active_segment, ACOUSTIC_FEATURE_NAMES, N_ACOUSTIC_FEATURES
from features.behavioral import behavioral_event_features, BEHAVIORAL_FEATURE_NAMES, N_BEHAVIORAL_FEATURES
from preprocessing.diarization import diarize_stereo
from model.density import BlockDensityModel
from model.calibration import StackingCalibrator, bayes_posterior, sigmoid
from model.llr import ScoreAccumulator

TARGET_SR = 16000  # todo el pipeline (features, entrenamiento, inferencia) trabaja a esta tasa
SEGMENT_SECONDS = 1.0  # tamaño del segmento acústico agregado (ver README)


def load_wav_any(path_or_bytes, target_sr: int = TARGET_SR):
    """Carga un WAV (path, bytes o file-like) a CUALQUIER sample rate de
    origen, lo resamplea siempre a target_sr (16 kHz por defecto) y
    regresa (signal, sr). Preserva estéreo si existe (para diarización).
    Esto garantiza que el modelo siempre "ve" audio a la misma tasa, sin
    importar a qué frecuencia se grabó originalmente."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(path_or_bytes), always_2d=False)
    else:
        data, sr = sf.read(path_or_bytes, always_2d=False)
    data = data.astype(np.float64)
    if sr != target_sr:
        if data.ndim == 1:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        else:
            data = np.stack([librosa.resample(data[:, c], orig_sr=sr, target_sr=target_sr)
                              for c in range(data.shape[1])], axis=1)
        sr = target_sr
    return data, sr


def split_channels(signal: np.ndarray):
    """Regresa (caller, agent_or_None)."""
    if signal.ndim == 1:
        return signal, None
    return signal[:, 0], signal[:, 1]


class VoiceAuthenticityDetector:
    """El "binario" del modelo: dos BlockDensityModel + un StackingCalibrator
    + metadatos (umbral, prior, versión). Serializable con joblib."""

    VERSION = "1.0"

    def __init__(self):
        self.acoustic_block = BlockDensityModel(ACOUSTIC_FEATURE_NAMES)
        self.behavioral_block = BlockDensityModel(BEHAVIORAL_FEATURE_NAMES)
        self.calibrator = StackingCalibrator()
        self.eta = 0.0            # umbral de decisión sobre el score LLR total
        self.prior_h1 = 0.3       # proporción esperada de voz sintética
        self.segment_seconds = SEGMENT_SECONDS
        self.train_report = {}

    # ---------- persistencia ----------
    def save(self, path: str):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> "VoiceAuthenticityDetector":
        return joblib.load(path)

    # ---------- inferencia ----------
    def new_session(self) -> "StreamingSession":
        return StreamingSession(self)

    def predict_offline(self, path_or_bytes, turns: list[dict] | None = None) -> dict:
        signal, sr = load_wav_any(path_or_bytes)
        caller, agent = split_channels(signal)

        if turns is None:
            if agent is not None:
                turns = diarize_stereo(signal, sr)
            else:
                turns = []

        session = self.new_session()
        session.feed_full_call(caller, sr, turns, realtime=False)
        return session.final_result()


@dataclass
class StreamingSession:
    """Sesión de detección incremental: procesa segmentos de audio y eventos
    de turno conforme llegan, manteniendo el score acumulado y su desglose.
    Es el objeto que consume tanto el simulador de tiempo real como (en un
    despliegue real) el pipeline de streaming de la llamada en vivo."""

    detector: VoiceAuthenticityDetector
    acc: ScoreAccumulator = field(default_factory=ScoreAccumulator)
    _audio_buffer: list = field(default_factory=list)
    _buffer_t0: float = 0.0
    _t_cursor: float = 0.0
    _running_peak: float = 1e-9  # pico de amplitud observado hasta ahora (causal)
    started_at: float = field(default_factory=time.time)

    # ---- entrada incremental de audio (caller, canal 0) ----
    def push_audio_chunk(self, chunk: np.ndarray, sr: int, ref_peak: float | None = None) -> dict | None:
        """Agrega audio del caller al buffer; cuando se acumula un segmento
        completo (segment_seconds) calcula x_a y devuelve el snapshot del
        score. Si aún no hay suficiente audio, regresa None.

        Los segmentos de silencio/ruido de fondo (ver `is_active_segment`)
        se saltan: no se computan features ni se suman al score, para que
        la distribución de segmentos vista aquí coincida con la usada en
        entrenamiento (`train/fit_densities.py`).

        `ref_peak`: pico de referencia para medir energía relativa. En modo
        offline (`feed_full_call`) se pasa el pico de la llamada completa. En
        modo streaming real no se conoce de antemano, así que se usa
        `_running_peak` (el pico observado hasta el momento), que se
        actualiza de forma causal con cada chunk."""
        if chunk.size:
            self._running_peak = max(self._running_peak, float(np.abs(chunk).max()))
        peak = ref_peak if ref_peak is not None else self._running_peak

        self._audio_buffer.append(chunk)
        buffered = np.concatenate(self._audio_buffer) if len(self._audio_buffer) > 1 else self._audio_buffer[0]
        seg_len = int(self.detector.segment_seconds * sr)
        snap = None
        while len(buffered) >= seg_len:
            seg = buffered[:seg_len]
            buffered = buffered[seg_len:]
            t = self._buffer_t0 + self.detector.segment_seconds
            self._buffer_t0 = t
            self._t_cursor = t
            if not is_active_segment(seg, peak):
                continue  # silencio: no aporta evidencia, se descarta
            x_a = segment_features(seg, sr)
            attribution = self.detector.acoustic_block.llr_attribution(x_a)
            snap = self.acc.add_acoustic(t, attribution)
        self._audio_buffer = [buffered] if len(buffered) else []
        return snap

    # ---- entrada incremental de eventos de turno ----
    def push_turn_event(self, t_event: float, x_b: np.ndarray, event_meta: dict | None = None) -> dict:
        attribution = self.detector.behavioral_block.llr_attribution(x_b)
        self._t_cursor = max(self._t_cursor, t_event)
        return self.acc.add_behavioral(t_event, attribution, event_meta)

    # ---- helper: alimenta una llamada completa (offline o simulada) ----
    def feed_full_call(self, caller: np.ndarray, sr: int, turns: list[dict], realtime: bool = False):
        seg_len = int(self.detector.segment_seconds * sr)
        n_segments = len(caller) // seg_len
        behavioral_events = behavioral_event_features(turns) if turns else []
        # offline: se conoce toda la llamada de antemano, así que se usa el
        # pico real de la llamada completa como referencia de energía (igual
        # que en entrenamiento), en vez del pico causal usado en streaming.
        ref_peak = float(np.abs(caller).max()) if caller.size else 1e-9

        be_idx = 0
        for i in range(n_segments):
            seg = caller[i * seg_len:(i + 1) * seg_len]
            t = (i + 1) * self.detector.segment_seconds
            self.push_audio_chunk(seg, sr, ref_peak=ref_peak)

            # intercalar eventos comportamentales que ya "sucedieron" a este tiempo
            while be_idx < len(behavioral_events) and behavioral_events[be_idx][0] <= t:
                t_ev, x_b = behavioral_events[be_idx]
                self.push_turn_event(t_ev, x_b)
                be_idx += 1
            if realtime:
                time.sleep(self.detector.segment_seconds)

        while be_idx < len(behavioral_events):
            t_ev, x_b = behavioral_events[be_idx]
            self.push_turn_event(t_ev, x_b)
            be_idx += 1

    # ---- resultado ----
    def current_snapshot(self) -> dict:
        det = self.detector
        score = self.acc.score_total
        proba_stacked = det.calibrator.predict_proba(self.acc.llr_acoustic_cum, self.acc.llr_behavioral_cum)
        proba_bayes = bayes_posterior(score, det.prior_h1)
        return {
            "t": self._t_cursor,
            "score_total": score,
            "llr_acoustic_cum": self.acc.llr_acoustic_cum,
            "llr_behavioral_cum": self.acc.llr_behavioral_cum,
            "n_acoustic_segments": self.acc.n_acoustic_segments,
            "n_behavioral_events": self.acc.n_behavioral_events,
            "confidence": proba_stacked,
            "confidence_bayes_fallback": proba_bayes,
            "is_synthetic": bool(score > det.eta),
            "eta": det.eta,
        }

    def final_result(self) -> dict:
        snap = self.current_snapshot()
        is_synth = snap["is_synthetic"]
        confidence = snap["confidence"] if is_synth else 1.0 - snap["confidence"]
        return {
            "is_synthetic": is_synth,
            "confidence": round(float(confidence), 4),
            "score_total": snap["score_total"],
            "llr_acoustic_cum": snap["llr_acoustic_cum"],
            "llr_behavioral_cum": snap["llr_behavioral_cum"],
            "n_acoustic_segments": snap["n_acoustic_segments"],
            "n_behavioral_events": snap["n_behavioral_events"],
            "eta": snap["eta"],
            "history": self.acc.history,
        }
