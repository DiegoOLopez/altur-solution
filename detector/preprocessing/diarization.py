"""
Diarización simple: dado audio estéreo (canal0=caller, canal1=agente), corre
VAD independiente en cada canal y arma la lista de turnos en el mismo
formato que el JSON de anotación manual:
    [{"channel": 0|1, "start": t0, "end": t1}, ...]

Esto permite entrenar/inferir aunque una llamada no traiga turnos.json —
se usan los turnos "reales" (provistos) cuando existen porque son más
precisos; si no, se cae aquí automáticamente.
"""
from __future__ import annotations

import numpy as np

from preprocessing.vad import vad_segments


def diarize_stereo(stereo: np.ndarray, sr: int) -> list[dict]:
    """stereo: array shape (n_samples, 2) o (2, n_samples)."""
    stereo = np.asarray(stereo)
    if stereo.ndim == 1:
        raise ValueError("diarize_stereo requiere audio estéreo (2 canales)")
    if stereo.shape[0] == 2 and stereo.shape[1] != 2:
        stereo = stereo.T  # normalizar a (n_samples, 2)

    caller = stereo[:, 0]
    agent = stereo[:, 1]

    turns = []
    for start, end in vad_segments(caller, sr):
        turns.append({"channel": 0, "start": float(start), "end": float(end)})
    for start, end in vad_segments(agent, sr):
        turns.append({"channel": 1, "start": float(start), "end": float(end)})

    turns.sort(key=lambda t: t["start"])
    return turns
