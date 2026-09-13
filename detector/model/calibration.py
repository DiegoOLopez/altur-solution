"""
Calibración (sección 1.5) y stacking logístico.

Dos capas, ambas interpretables:

1) Calibración bayesiana directa del LLR total con prior:
       p(H1|x) = sigmoid(score(x) + log(p(H1)/p(H0)))
   Útil como base / fallback y para razonar sobre el umbral eta vía ROC.

2) Stacking logístico (lo que pidió el usuario): en vez de sumar
   ciegamente LLR_acustico + LLR_comportamental con peso 1.0 cada uno (que
   asume que ambos bloques son igual de confiables), se ajusta una
   regresión logística de 2 entradas:

       p(H1|x) = sigmoid(w_a * LLR_acustico_total + w_b * LLR_comportamental_total + b)

   Sigue siendo totalmente interpretable (2 pesos + 1 bias, uno por bloque),
   pero permite que el entrenamiento aprenda cuánto pesar cada bloque según
   qué tan informativo resultó ser en los datos reales, y actúa a la vez
   como calibrador de probabilidad (Platt scaling sobre los dos scores).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def sigmoid(x: float) -> float:
    x = np.clip(x, -500, 500)
    return float(1.0 / (1.0 + np.exp(-x)))


def bayes_posterior(score: float, prior_h1: float) -> float:
    log_prior_odds = np.log(prior_h1 / (1 - prior_h1))
    return sigmoid(score + log_prior_odds)


class StackingCalibrator:
    """Regresión logística de 2 features: [LLR_acustico_total, LLR_comportamental_total].

    Los LLR crudos acumulados sobre una llamada completa pueden llegar a ser
    de magnitud grande (suma de muchos términos gaussianos), lo que satura
    una logística sin más — por eso se estandariza (z-score) cada bloque con
    la media/desviación observadas en entrenamiento antes de ajustar los
    pesos. Los pesos resultantes siguen siendo interpretables: representan
    el efecto de una desviación estándar de evidencia de ese bloque sobre
    el log-odds final. Se usa C pequeño (más regularización) porque con
    pocas llamadas y clases separables la logística sin penalizar diverge."""

    def __init__(self, C: float = 0.5):
        self.C = C
        self.clf = LogisticRegression(C=C, max_iter=1000)
        self.mean_ = np.zeros(2)
        self.std_ = np.ones(2)
        self.fitted = False

    def _standardize(self, llr_acoustic, llr_behavioral):
        x = np.array([llr_acoustic, llr_behavioral], dtype=np.float64)
        return (x - self.mean_) / self.std_

    def fit(self, llr_acoustic: np.ndarray, llr_behavioral: np.ndarray, y: np.ndarray) -> "StackingCalibrator":
        X = np.stack([llr_acoustic, llr_behavioral], axis=1)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), 1e-6)
        Xz = (X - self.mean_) / self.std_
        self.clf.fit(Xz, y)
        self.fitted = True
        return self

    def predict_proba(self, llr_acoustic: float, llr_behavioral: float) -> float:
        if not self.fitted:
            # fallback: suma simple + sigmoide (sección 1.5) si no hay
            # suficientes datos para entrenar el stacking
            return sigmoid(llr_acoustic + llr_behavioral)
        xz = self._standardize(llr_acoustic, llr_behavioral).reshape(1, -1)
        return float(self.clf.predict_proba(xz)[0, 1])

    @property
    def weights(self) -> dict:
        if not self.fitted:
            return {"w_acoustic": 1.0, "w_behavioral": 1.0, "bias": 0.0}
        w = self.clf.coef_[0]
        return {"w_acoustic": float(w[0]), "w_behavioral": float(w[1]), "bias": float(self.clf.intercept_[0]),
                "note": "pesos sobre LLR estandarizado (z-score); ver mean_/std_ para volver a escala cruda"}

    def to_dict(self) -> dict:
        if not self.fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "C": self.C,
            "coef": self.clf.coef_.tolist(),
            "intercept": self.clf.intercept_.tolist(),
            "classes": self.clf.classes_.tolist(),
            "mean_": self.mean_.tolist(),
            "std_": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StackingCalibrator":
        obj = cls(C=d.get("C", 0.5))
        if d.get("fitted"):
            obj.clf.coef_ = np.array(d["coef"])
            obj.clf.intercept_ = np.array(d["intercept"])
            obj.clf.classes_ = np.array(d["classes"])
            obj.mean_ = np.array(d["mean_"])
            obj.std_ = np.array(d["std_"])
            obj.fitted = True
        return obj


def choose_threshold_youden(scores: np.ndarray, y: np.ndarray) -> float:
    """Elige eta maximizando el índice de Youden (TPR - FPR) sobre la curva
    ROC sacada del set de validación, sin reentrenar ninguna densidad
    (sección 1.5)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])
