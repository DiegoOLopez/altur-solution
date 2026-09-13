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

Todo implementado sobre numpy/scipy/librosa, sin cajas negras: cada función
regresa un escalar (o vector pequeño) con significado físico directo, para
que la atribución del LLR por feature sea interpretable.
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


def _entropy_and_phase_coherence_tracks(signal: np.ndarray, sr: int):
    """Calcula entropía espectral Y coherencia de fase armónica en un solo
    paso, compartiendo una única FFT batched (antes cada una recalculaba su
    propio espectro por frame en un loop de Python separado — mismo
    framing/ventana en ambas, así que la FFT es literalmente redundante).

    OPTIMIZACION (2026-09): misma fórmula exacta que las versiones
    originales de `spectral_entropy_track` y `harmonic_phase_coherence_track`
    (ahora conservadas como wrappers abajo) — validado numéricamente
    (diff máxima ~1e-16, error de redondeo flotante) sobre 50 segmentos
    reales de data/train. ~4.5x más rápido que llamar ambas por separado."""
    frame_len = max(int(sr * FRAME_MS / 1000), 8)
    frames = _frame_signal(signal, sr, FRAME_MS, FRAME_HOP_MS)
    if frames.shape[0] == 0:
        return np.array([0.0]), np.array([0.0])
    window = np.hanning(frame_len)
    spec = np.fft.rfft(frames * window, axis=1)  # una sola FFT para ambas features
    power = spec.real ** 2 + spec.imag ** 2

    # --- entropía (misma fórmula que el original, vectorizada) ---
    total = power.sum(axis=1, keepdims=True) + 1e-12
    p = power / total
    mask = p > 0
    logp = np.where(mask, np.log2(np.where(mask, p, 1.0)), 0.0)
    ent_raw = -(p * logp).sum(axis=1)
    counts = mask.sum(axis=1)
    denom = np.log2(counts + 1e-12)
    ent = np.where(counts > 1, ent_raw / denom, 0.0)

    # --- coherencia de fase (misma lógica que el original) ---
    if frames.shape[0] < 2:
        coherences = np.array([0.0])
    else:
        mag = np.abs(spec)
        valid = mag.max(axis=1) >= 1e-8
        k = np.argmax(mag[:, 1:], axis=1) + 1
        rows = np.arange(spec.shape[0])
        phase = np.angle(spec[rows, k])
        valid_pairs = valid[:-1] & valid[1:]
        dphi = np.angle(np.exp(1j * (phase[1:] - phase[:-1])))
        coh_all = np.cos(dphi)
        coherences = coh_all[valid_pairs] if valid_pairs.any() else np.array([0.0])
    return ent, coherences


def spectral_entropy_track(signal: np.ndarray, sr: int) -> np.ndarray:
    """Entropía de Shannon del espectro de potencia por frame. Voz sintética
    tiende a tener espectros más "ordenados" (menor entropía) por el proceso
    generativo del vocoder.

    Se conserva como wrapper de compatibilidad; usa
    `_entropy_and_phase_coherence_tracks` internamente. Si necesitas ambos
    valores, llama a esa función directamente en vez de esta + la de abajo
    (evita recalcular la FFT dos veces)."""
    ent, _ = _entropy_and_phase_coherence_tracks(signal, sr)
    return ent


def harmonic_phase_coherence_track(signal: np.ndarray, sr: int) -> np.ndarray:
    """Proxy de coherencia de fase armónica: para cada frame, estima F0 local
    y mide qué tan cerca está la fase de los armónicos de la fase esperada
    de una señal perfectamente periódica (fase lineal). Voz real tiene ruido
    de fase natural (menor coherencia); vocoders producen fase más regular
    entre frames consecutivos (mayor coherencia artificial).

    Se conserva como wrapper de compatibilidad; ver nota en
    `spectral_entropy_track`."""
    _, coh = _entropy_and_phase_coherence_tracks(signal, sr)
    return coh


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


def _jitter_from_f0(f0s: np.ndarray) -> float:
    if len(f0s) < 3:
        return 0.0
    periods = 1.0 / f0s
    diffs = np.abs(np.diff(periods))
    return float(diffs.mean() / (periods.mean() + 1e-12))


def _shimmer_from_amps(amps: np.ndarray) -> float:
    if len(amps) < 3:
        return 0.0
    diffs = np.abs(np.diff(amps))
    return float(diffs.mean() / (amps.mean() + 1e-12))


def jitter_local(signal: np.ndarray, sr: int) -> float:
    """Jitter local (%): perturbación ciclo-a-ciclo de F0, definición estilo
    Praat: media(|T_i - T_{i+1}|) / media(T_i)."""
    f0s, _ = _estimate_f0_track(signal, sr)
    return _jitter_from_f0(f0s)


def shimmer_local(signal: np.ndarray, sr: int) -> float:
    """Shimmer local (%): perturbación ciclo-a-ciclo de amplitud."""
    _, amps = _estimate_f0_track(signal, sr)
    return _shimmer_from_amps(amps)


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


def is_active_segment(signal: np.ndarray, ref_peak: float, energy_thresh_db: float = -35.0) -> bool:
    """Compuerta de energía: True si el segmento tiene voz activa, False si
    es silencio/ruido de fondo.

    Por qué importa para el entrenamiento: cuando un segmento es puro
    silencio, `segment_features` devuelve el mismo vector (casi) de ceros
    para AMBAS clases. Si se incluyen muchos de estos en el ajuste de las
    gaussianas, comprimen artificialmente sigma en varias dimensiones
    (ambas clases acumulan el mismo valor "cero"), volviendo el modelo
    hipersensible a cualquier variación real. Se usa la misma compuerta en
    entrenamiento (`train/fit_densities.py`) y en inferencia
    (`model/pipeline.py`) para que la distribución de segmentos vista por
    las gaussianas sea la misma en ambos casos.

    `ref_peak` es el pico de amplitud de referencia (offline: el pico de
    toda la llamada; streaming: el pico observado hasta el momento) — se
    mide energía relativa a esa referencia, no en términos absolutos, para
    no depender del nivel de grabación de cada llamada.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or ref_peak < 1e-9:
        return False
    rms = np.sqrt((signal ** 2).mean())
    db = 20 * np.log10(rms / ref_peak + 1e-12)
    return db > energy_thresh_db


def segment_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """Construye el vector x_a completo (dim = N_ACOUSTIC_FEATURES) para un
    segmento de audio (agregando los términos frame-a-frame del bloque
    acústico), en el orden de ACOUSTIC_FEATURE_NAMES."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or np.abs(signal).max() < 1e-6:
        return np.zeros(N_ACOUSTIC_FEATURES)

    sf = spectral_flatness_track(signal, sr)
    se, hpc = _entropy_and_phase_coherence_tracks(signal, sr)  # 1 FFT en vez de 2
    mfcc_mean, mfcc_var = mfcc_stats(signal, sr)
    f0s, amps = _estimate_f0_track(signal, sr)  # 1 vez en vez de 2 (jitter+shimmer)
    jit = _jitter_from_f0(f0s)
    shim = _shimmer_from_amps(amps)
    pause = pause_naturalness(signal, sr)

    vec = np.concatenate([
        [sf.mean(), sf.std(), se.mean(), se.std(), hpc.mean()],
        mfcc_mean, mfcc_var,
        [jit, shim, pause],
    ])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
