"""
Modelo de densidad por bloque: gaussiana con covarianza diagonal
(naive Bayes gaussiano), sección 1.3 del documento de diseño.

    log p(x|H) = sum_i [ -1/2 log(2*pi*sigma_i^2) - (x_i - mu_i)^2 / (2*sigma_i^2) ]

"""
from __future__ import annotations

import numpy as np

MIN_VAR = 1e-6  # piso de varianza para evitar log(0) / división por cero


class DiagonalGaussian:
    """Gaussiana con covarianza diagonal para un bloque de features bajo una
    hipótesis (H0 o H1)."""

    def __init__(self, mean: np.ndarray | None = None, var: np.ndarray | None = None):
        self.mean = mean
        self.var = var

    def fit(self, X: np.ndarray) -> "DiagonalGaussian":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.var = np.maximum(X.var(axis=0), MIN_VAR)
        return self

    def log_prob_terms(self, x: np.ndarray) -> np.ndarray:
        """log p(x|H) por dimensión (para atribución exacta por feature)."""
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.log(2 * np.pi * self.var) - (x - self.mean) ** 2 / (2 * self.var)

    def log_prob(self, x: np.ndarray) -> float:
        return float(self.log_prob_terms(x).sum())

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "DiagonalGaussian":
        return cls(mean=np.array(d["mean"]), var=np.array(d["var"]))


class BlockDensityModel:
    """Par de gaussianas diagonales (H0=humano, H1=sintético) para un bloque
    de evidencia (acústico o comportamental), con cómputo de LLR y
    atribución exacta por feature."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.h0 = DiagonalGaussian()
        self.h1 = DiagonalGaussian()

    def fit(self, X_h0: np.ndarray, X_h1: np.ndarray) -> "BlockDensityModel":
        self.h0.fit(X_h0)
        self.h1.fit(X_h1)
        return self

    def llr(self, x: np.ndarray) -> float:
        return self.h1.log_prob(x) - self.h0.log_prob(x)

    def llr_attribution(self, x: np.ndarray) -> dict:
        """LLR total + contribución exacta de cada feature (h1_term - h0_term)."""
        t1 = self.h1.log_prob_terms(x)
        t0 = self.h0.log_prob_terms(x)
        per_feature = (t1 - t0)
        return {
            "llr_total": float(per_feature.sum()),
            "per_feature": {name: float(v) for name, v in zip(self.feature_names, per_feature)},
        }

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "h0": self.h0.to_dict(),
            "h1": self.h1.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockDensityModel":
        m = cls(d["feature_names"])
        m.h0 = DiagonalGaussian.from_dict(d["h0"])
        m.h1 = DiagonalGaussian.from_dict(d["h1"])
        return m
