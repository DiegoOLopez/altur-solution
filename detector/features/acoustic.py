"""
Bloque acústico (x_a) — canal 0, nivel frame(25ms) -> agregado por segmento.

Implementa exactamente las features listadas en el documento de diseño:
  - Planitud espectral (spectral flatness)
  - Entropía espectral
  - MFCCs (media + varianza por segmento)
  - Coherencia de fase armónica
  - Jitter (perturbación de F0)
  - Shimmer (perturbación de amplitud)
  - Naturalidad de pausas intra-frase

"""
from __future__ import annotations

import numpy as np
import librosa

FRAME_MS = 25
FRAME_HOP_MS = 10
N_MFCC = 13

# Nombres de las dimensiones del vector x_a, en el orden exacto en que se
# construyen (ver `segment_features`). Se usan en todo el sistema para
# etiquetar la atribución por feature en el dashboard.
ACOUSTIC_FEATURE_NAMES = (
    ["spectral_flatness_mean", "spectral_flatness_std",
     "spectral_entropy_mean", "spectral_entropy_std",
     "harmonic_phase_coherence_mean"]
    + [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_var" for i in range(1, N_MFCC + 1)]
    + ["jitter_local", "shimmer_local", "pause_naturalness"]
)
N_ACOUSTIC_FEATURES = len(ACOUSTIC_FEATURE_NAMES)


def _frame_signal(signal: np.ndarray, sr: int, frame_ms=FRAME_MS, hop_ms=FRAME_HOP_MS):
    frame_len = max(int(sr * frame_ms / 1000), 8)
    hop_len = max(int(sr * hop_ms / 1000), 4)
    if len(signal) < frame_len:
        return np.empty((0, frame_len))
    return librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T


def spectral_flatness_track(signal: np.ndarray, sr: int) -> np.ndarray:
    """Planitud espectral por frame de 25ms (vocoders neuronales dejan huecos
    espectrales en altas frecuencias -> planitud anómala)."""
    frame_len = max(int(sr * FRAME_MS / 1000), 8)
    hop_len = max(int(sr * FRAME_HOP_MS / 1000), 4)
    if len(signal) < frame_len:
        return np.array([0.0])
    sf = librosa.feature.spectral_flatness(y=signal, n_fft=frame_len, hop_length=hop_len)
    return sf.flatten()


def spectral_entropy_track(signal: np.ndarray, sr: int) -> np.ndarray:
    """Entropía de Shannon del espectro de potencia por frame. Voz sintética
    tiende a tener espectros más "ordenados" (menor entropía) por el proceso
    generativo del vocoder."""
    frame_len = max(int(sr * FRAME_MS / 1000), 8)
    hop_len = max(int(sr * FRAME_HOP_MS / 1000), 4)
    frames = _frame_signal(signal, sr, FRAME_MS, FRAME_HOP_MS)
    if frames.shape[0] == 0:
        return np.array([0.0])
    window = np.hanning(frame_len)
    ent = np.empty(frames.shape[0])
    for i, f in enumerate(frames):
        spec = np.abs(np.fft.rfft(f * window)) ** 2
        p = spec / (spec.sum() + 1e-12)
        p = p[p > 0]
        ent[i] = -(p * np.log2(p)).sum() / np.log2(len(p) + 1e-12) if len(p) > 1 else 0.0
    return ent


def harmonic_phase_coherence_track(signal: np.ndarray, sr: int) -> np.ndarray:
    """Proxy de coherencia de fase armónica: para cada frame, estima F0 local
    y mide qué tan cerca está la fase de los armónicos de la fase esperada
    de una señal perfectamente periódica (fase lineal). Voz real tiene ruido
    de fase natural (menor coherencia); vocoders producen fase más regular
    entre frames consecutivos (mayor coherencia artificial)."""
    frame_len = max(int(sr * FRAME_MS / 1000), 8)
    hop_len = max(int(sr * FRAME_HOP_MS / 1000), 4)
    frames = _frame_signal(signal, sr, FRAME_MS, FRAME_HOP_MS)
    if frames.shape[0] < 2:
        return np.array([0.0])
    window = np.hanning(frame_len)
    phases = []
    for f in frames:
        spec = np.fft.rfft(f * window)
        mag = np.abs(spec)
        if mag.max() < 1e-8:
            phases.append(None)
            continue
        k = int(np.argmax(mag[1:])) + 1  # bin fundamental dominante
        phases.append(np.angle(spec[k]))
    coherences = []
    for a, b in zip(phases[:-1], phases[1:]):
        if a is None or b is None:
            continue
        dphi = np.angle(np.exp(1j * (b - a)))  # diferencia envuelta a [-pi,pi]
        coherences.append(np.cos(dphi))  # 1 = fase perfectamente coherente
    return np.array(coherences) if coherences else np.array([0.0])


def mfcc_stats(signal: np.ndarray, sr: int, n_mfcc: int = N_MFCC):
    """Media y varianza de MFCCs sobre el segmento completo (fingerprint del
    tracto vocal / modelo generador)."""
    frame_len = max(int(sr * FRAME_MS / 1000), 8)
    hop_len = max(int(sr * FRAME_HOP_MS / 1000), 4)
    if len(signal) < frame_len:
        return np.zeros(n_mfcc), np.zeros(n_mfcc)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=frame_len, hop_length=hop_len,
                                 n_mels=26, fmax=sr / 2)
    return mfcc.mean(axis=1), mfcc.var(axis=1)


def _estimate_f0_track(signal: np.ndarray, sr: int, fmin=60.0, fmax=400.0):
    """F0 por autocorrelación, frame a frame (requiere varios ciclos glotales
    por frame -> se usa una ventana algo más larga que 25ms, 40ms, tal como
    indica el documento para jitter/shimmer)."""
    frame_len = max(int(sr * 0.04), 16)
    hop_len = max(int(sr * 0.02), 8)
    frames = _frame_signal(signal, sr, frame_ms=40, hop_ms=20)
    if frames.shape[0] == 0:
        return np.array([]), np.array([])
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    f0s, amps = [], []
    for f in frames:
        f = f - f.mean()
        energy = np.sqrt((f ** 2).mean())
        if energy < 1e-4:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        ac = ac / (ac[0] + 1e-12)
        if lag_max >= len(ac):
            continue
        seg = ac[lag_min:lag_max]
        if len(seg) == 0:
            continue
        peak = np.argmax(seg) + lag_min
        if ac[peak] < 0.3:  # frame no suficientemente periódico (no sonoro)
            continue
        f0 = sr / peak
        f0s.append(f0)
        amps.append(energy)
    return np.array(f0s), np.array(amps)


def jitter_local(signal: np.ndarray, sr: int) -> float:
    """Jitter local (%): perturbación ciclo-a-ciclo de F0, definición estilo
    Praat: media(|T_i - T_{i+1}|) / media(T_i)."""
    f0s, _ = _estimate_f0_track(signal, sr)
    if len(f0s) < 3:
        return 0.0
    periods = 1.0 / f0s
    diffs = np.abs(np.diff(periods))
    return float(diffs.mean() / (periods.mean() + 1e-12))


def shimmer_local(signal: np.ndarray, sr: int) -> float:
    """Shimmer local (%): perturbación ciclo-a-ciclo de amplitud."""
    _, amps = _estimate_f0_track(signal, sr)
    if len(amps) < 3:
        return 0.0
    diffs = np.abs(np.diff(amps))
    return float(diffs.mean() / (amps.mean() + 1e-12))


def pause_naturalness(signal: np.ndarray, sr: int, top_db: float = 30.0) -> float:
    """Naturalidad de las micro-pausas intra-frase (respiración). Se mide
    como el coeficiente de variación de las duraciones de silencio interno
    del segmento: la respiración humana produce pausas de duración variable;
    el silencio "de librería" de un TTS concatenativo/neuronal tiende a ser
    más uniforme (CV bajo) o directamente inexistente."""
    intervals = librosa.effects.split(signal, top_db=top_db)
    if len(intervals) < 2:
        return 0.0
    gaps = (intervals[1:, 0] - intervals[:-1, 1]) / sr
    gaps = gaps[gaps > 0.01]
    if len(gaps) < 2:
        return 0.0
    return float(gaps.std() / (gaps.mean() + 1e-12))


def segment_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """Construye el vector x_a completo (dim = N_ACOUSTIC_FEATURES) para un
    segmento de audio (agregando los términos frame-a-frame del bloque
    acústico), en el orden de ACOUSTIC_FEATURE_NAMES."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or np.abs(signal).max() < 1e-6:
        return np.zeros(N_ACOUSTIC_FEATURES)

    sf = spectral_flatness_track(signal, sr)
    se = spectral_entropy_track(signal, sr)
    hpc = harmonic_phase_coherence_track(signal, sr)
    mfcc_mean, mfcc_var = mfcc_stats(signal, sr)
    jit = jitter_local(signal, sr)
    shim = shimmer_local(signal, sr)
    pause = pause_naturalness(signal, sr)

    vec = np.concatenate([
        [sf.mean(), sf.std(), se.mean(), se.std(), hpc.mean()],
        mfcc_mean, mfcc_var,
        [jit, shim, pause],
    ])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
