"""
Cómputo de LLR acumulado (sección 1.4 del documento):

    score(t) = sum_{k<=t} LLR_acustico(x_a^(k)) + sum_{eventos<=t} LLR_comportamental(x_b)

Aditividad exacta entre bloques (independencia condicional asumida dado H),
por lo que cada término del score tiene atribución exacta a un bloque y a
una feature dentro de ese bloque — no es una aproximación tipo SHAP.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreAccumulator:
    """Acumulador incremental del LLR total, con desglose por bloque y por
    feature, listo para alimentar tanto la decisión final como el dashboard
    en vivo (curva score(t) apilada por bloque)."""

    llr_acoustic_cum: float = 0.0
    llr_behavioral_cum: float = 0.0
    n_acoustic_segments: int = 0
    n_behavioral_events: int = 0
    history: list = field(default_factory=list)  # snapshots para el dashboard

    @property
    def score_total(self) -> float:
        return self.llr_acoustic_cum + self.llr_behavioral_cum

    def add_acoustic(self, t: float, attribution: dict) -> dict:
        self.llr_acoustic_cum += attribution["llr_total"]
        self.n_acoustic_segments += 1
        snap = self._snapshot(t, block="acoustic", attribution=attribution)
        self.history.append(snap)
        return snap

    def add_behavioral(self, t: float, attribution: dict, event_meta: dict | None = None) -> dict:
        self.llr_behavioral_cum += attribution["llr_total"]
        self.n_behavioral_events += 1
        snap = self._snapshot(t, block="behavioral", attribution=attribution, event_meta=event_meta)
        self.history.append(snap)
        return snap

    def _snapshot(self, t: float, block: str, attribution: dict, event_meta: dict | None = None) -> dict:
        return {
            "t": t,
            "block": block,
            "llr_increment": attribution["llr_total"],
            "per_feature": attribution["per_feature"],
            "llr_acoustic_cum": self.llr_acoustic_cum,
            "llr_behavioral_cum": self.llr_behavioral_cum,
            "score_cum": self.score_total,
            "event_meta": event_meta,
        }
