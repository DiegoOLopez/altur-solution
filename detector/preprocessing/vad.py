"""
VAD (Voice Activity Detection) por canal.

Implementación ligera basada en energía RMS + cruce por cero, con histéresis,
suficiente para audio a 16 kHz y para funcionar en tiempo real (sin modelos
pesados, latencia mínima). Se usa solo cuando la llamada no trae ya un JSON
de turnos anotado (como el que proveyó Altur) — si ya existen turnos, este
módulo no es necesario.
"""
from __future__ import annotations

import numpy as np


def frame_generator(signal: np.ndarray, sr: int, frame_ms: int = 20):
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(signal) // frame_len
    for i in range(n_frames):
        yield i * frame_len / sr, signal[i * frame_len:(i + 1) * frame_len]


def vad_segments(signal: np.ndarray, sr: int, frame_ms: int = 20,
                  energy_thresh_db: float = -40.0,
                  min_speech_s: float = 0.15,
                  min_silence_s: float = 0.12) -> list[tuple[float, float]]:
    """Regresa lista de (start, end) en segundos donde hay voz activa."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return []
    ref = np.abs(signal).max() + 1e-9

    frames = list(frame_generator(signal, sr, frame_ms))
    if not frames:
        return []

    is_speech = []
    for t, f in frames:
        if len(f) == 0:
            is_speech.append(False)
            continue
        rms = np.sqrt((f ** 2).mean()) / ref
        db = 20 * np.log10(rms + 1e-12)
        is_speech.append(db > energy_thresh_db)

    frame_dur = frame_ms / 1000
    min_speech_frames = max(int(min_speech_s / frame_dur), 1)
    min_silence_frames = max(int(min_silence_s / frame_dur), 1)

    # cerrar huecos cortos de silencio dentro de habla (histéresis)
    smoothed = list(is_speech)
    i = 0
    while i < len(smoothed):
        if not smoothed[i]:
            j = i
            while j < len(smoothed) and not smoothed[j]:
                j += 1
            if (j - i) < min_silence_frames and 0 < i and j < len(smoothed):
                for k in range(i, j):
                    smoothed[k] = True
            i = j
        else:
            i += 1

    # extraer segmentos contiguos de habla, descartando los muy cortos
    segments = []
    i = 0
    while i < len(smoothed):
        if smoothed[i]:
            j = i
            while j < len(smoothed) and smoothed[j]:
                j += 1
            if (j - i) >= min_speech_frames:
                start = frames[i][0]
                end = frames[j - 1][0] + frame_dur
                segments.append((start, end))
            i = j
        else:
            i += 1
    return segments
