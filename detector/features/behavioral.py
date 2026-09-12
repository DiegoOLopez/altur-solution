"""
Bloque comportamental (x_b) — ambos canales, nivel evento de turno.

Clasifica eventos de turno (interrupción, silencio, solape) a partir de una
lista de turnos {channel, start, end} (canal 0 = caller, canal 1 = agente) y
extrae, para cada evento, un vector de evidencia comportamental que se puede
sumar al score de forma incremental conforme ocurren los eventos (sección
1.4 del documento: si no ha ocurrido evento, no se imputa nada).

Features (en el orden de BEHAVIORAL_FEATURE_NAMES):
  - reaction_latency:            latencia de reacción tras el turno del agente
  - recovery_latency_running_var: varianza *incremental* (Welford) de las
                                   latencias observadas hasta este evento en
                                   la llamada -> "recuperación consistente es
                                   señal de máquina" (baja varianza = H1)
  - is_overlap:                  1 si el caller interrumpió/solapó al agente
                                   (latencia negativa), 0 si fue tras silencio
  - backchannel_present:         1 si hubo un backchannel corto del caller
                                   durante el turno del agente, 0 si no
  - backchannel_timing:          offset normalizado del backchannel dentro
                                   del turno del agente (0=inicio, 1=fin);
                                   0.5 (neutro) si no hubo backchannel
"""
from __future__ import annotations

import numpy as np

BEHAVIORAL_FEATURE_NAMES = [
    "reaction_latency",
    "recovery_latency_running_var",
    "is_overlap",
    "backchannel_present",
    "backchannel_timing",
]
N_BEHAVIORAL_FEATURES = len(BEHAVIORAL_FEATURE_NAMES)

BACKCHANNEL_MAX_DUR = 0.6  # s: duración típica de un "ajá"/"sí"


class WelfordVariance:
    """Varianza incremental exacta (Welford) — permite que la feature
    'recovery_latency_running_var' se calcule online, sin re-escanear toda
    la llamada en cada evento (coherente con la actualización online del
    documento, sección 1.6)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> float:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        return self.variance

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)


def sort_turns(turns: list[dict]) -> list[dict]:
    return sorted(turns, key=lambda t: t["start"])


def merge_same_channel_turns(turns: list[dict], max_gap_s: float = 0.5) -> list[dict]:
    """Un VAD/diarizador real suele fragmentar una sola intervención en
    varios turnos consecutivos del mismo canal separados por micro-pausas.
    Para no contar cada fragmento como una "reacción" distinta, se fusionan
    turnos consecutivos del mismo canal si el hueco entre ellos es menor a
    `max_gap_s` y no hay ningún turno del otro canal en medio."""
    turns = sort_turns(turns)
    if not turns:
        return []
    merged = [dict(turns[0])]
    for t in turns[1:]:
        last = merged[-1]
        if t["channel"] == last["channel"] and t["start"] - last["end"] <= max_gap_s:
            last["end"] = max(last["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def classify_turn_events(turns: list[dict]) -> list[dict]:
    """A partir de la lista cruda de turnos, construye la secuencia de
    'eventos de turno' relevantes para el bloque comportamental: cada vez
    que un turno de agente (channel=1) es seguido por un turno de caller
    (channel=0), se genera un evento con su latencia de reacción (negativa
    si hubo solape/interrupción) y se detectan backchannels del caller
    ocurridos *durante* el turno del agente.

    Regresa una lista de dicts:
      {t_event, latency, is_overlap, backchannel_present, backchannel_timing}
    ordenada temporalmente, lista para alimentar `behavioral_event_features`.
    """
    turns = merge_same_channel_turns(turns)
    agent_turns = [t for t in turns if t["channel"] == 1]
    caller_turns = [t for t in turns if t["channel"] == 0]
    used = set()  # índices de caller_turns ya consumidos (evita doble conteo)

    events = []
    for agent_turn in agent_turns:
        a_start, a_end = agent_turn["start"], agent_turn["end"]

        # backchannels: turnos cortos del caller, no usados aún, que ocurren
        # *dentro* de la ventana temporal del turno del agente
        backchannel = None
        backchannel_idx = None
        for idx, ct in enumerate(caller_turns):
            if idx in used:
                continue
            dur = ct["end"] - ct["start"]
            if dur <= BACKCHANNEL_MAX_DUR and a_start <= ct["start"] <= a_end:
                backchannel, backchannel_idx = ct, idx
                break
        if backchannel_idx is not None:
            used.add(backchannel_idx)

        # siguiente turno del caller (no usado aún) que empieza en o después
        # del inicio del turno del agente (posible interrupción) o después
        # de que termine (reacción normal)
        next_caller, next_idx = None, None
        for idx, ct in enumerate(caller_turns):
            if idx in used:
                continue
            if ct["start"] >= a_start:
                next_caller, next_idx = ct, idx
                break
        if next_caller is None:
            continue
        used.add(next_idx)

        latency = next_caller["start"] - a_end  # negativo = solape/interrupción
        events.append({
            "t_event": max(next_caller["start"], a_end),
            "latency": latency,
            "is_overlap": 1.0 if latency < 0 else 0.0,
            "backchannel_present": 1.0 if backchannel is not None else 0.0,
            "backchannel_timing": (
                (backchannel["start"] - a_start) / max(a_end - a_start, 1e-6)
                if backchannel is not None else 0.5
            ),
        })

    events.sort(key=lambda e: e["t_event"])
    return events


def behavioral_event_features(turns: list[dict]) -> list[tuple[float, np.ndarray]]:
    """Regresa lista de (t_event, x_b) en orden temporal, usando varianza
    incremental para 'recovery_latency_running_var'."""
    events = classify_turn_events(turns)
    var_tracker = WelfordVariance()
    out = []
    for ev in events:
        running_var = var_tracker.update(ev["latency"])
        x_b = np.array([
            ev["latency"],
            running_var,
            ev["is_overlap"],
            ev["backchannel_present"],
            ev["backchannel_timing"],
        ], dtype=np.float64)
        out.append((ev["t_event"], x_b))
    return out
