"""
Métricas adicionales para juzgar qué tan confiable (no solo qué tan
"acertado") es el modelo, y para cuantificar la incertidumbre de las
métricas dado que el dataset es de unos cientos de llamadas.

- Brier score / ECE: la salida del modelo no es solo "sintético o no",
  es una CONFIANZA en [0,1] (ver `model/calibration.py`). AUC y accuracy
  no dicen nada sobre si esa confianza está bien calibrada (si el modelo
  dice "80% de confianza" ¿de verdad acierta ~80% de las veces en ese
  rango?). Son baratas de calcular y muy informativas para un sistema que
  se vende por su confianza reportada, no solo por su veredicto binario.

- Bootstrap CI sobre AUC: con pocas llamadas, un único número de AUC
  esconde cuánto podría moverse si el dataset hubiera sido ligeramente
  distinto. Un intervalo de confianza por bootstrap (remuestreo POR
  LLAMADA, no por segmento) es una forma barata y honesta de comunicar esa
  incertidumbre junto con la media ± std entre repeticiones de CV.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """ECE clásico: agrupa las predicciones en `n_bins` de confianza y
    compara, en cada bin, la confianza media contra la tasa real de
    aciertos (H1). Regresa también los datos por bin para poder graficar
    un diagrama de confiabilidad."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_report = []
    n = len(y)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            bins_report.append({"lo": float(lo), "hi": float(hi), "count": 0,
                                 "mean_confidence": None, "empirical_rate": None})
            continue
        mean_conf = float(p[mask].mean())
        emp_rate = float(y[mask].mean())
        ece += (count / n) * abs(mean_conf - emp_rate)
        bins_report.append({"lo": float(lo), "hi": float(hi), "count": count,
                             "mean_confidence": mean_conf, "empirical_rate": emp_rate})
    return {"ece": float(ece), "bins": bins_report}


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """IC por bootstrap (remuestreo con reemplazo POR LLAMADA) sobre el AUC.
    Si alguna clase queda ausente en una remuestra, esa iteración se
    descarta (no cuenta como AUC=0.5, para no sesgar el intervalo)."""
    y = np.asarray(y)
    score = np.asarray(score)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = y[idx], score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_b, s_b))
        except ValueError:
            continue
    if not aucs:
        return {"n_valid_boot": 0, "ci_low": None, "ci_high": None, "median": None}
    aucs = np.array(aucs)
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return {"n_valid_boot": int(len(aucs)), "ci_low": lo, "ci_high": hi, "median": float(np.median(aucs))}
