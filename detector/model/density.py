"""
Modelo de densidad por bloque: gaussiana con covarianza diagonal
(naive Bayes gaussiano), sección 1.3 del documento de diseño.

    log p(x|H) = sum_i [ -1/2 log(2*pi*sigma_i^2) - (x_i - mu_i)^2 / (2*sigma_i^2) ]

Por qué diagonal: Sigma^-1 se calcula una sola vez en entrenamiento (nunca en
el loop de inferencia), es más estable con pocos cientos de llamadas, y deja
que cada feature aporte su propio término al log-likelihood — la unidad
mínima de interpretabilidad del sistema.

--- Robustez con datasets pequeños (pocas llamadas, muchos segmentos) ---

Los segmentos/eventos de una misma llamada NO son muestras independientes
(mismo hablante, mismo canal). Si se estima sigma_i^2 agrupando todos los
segmentos de todas las llamadas como si fueran i.i.d. (MLE ingenuo), la
varianza queda sistemáticamente SUBESTIMADA: el modelo "cree" tener miles de
muestras independientes cuando en realidad solo tiene un puñado de llamadas
reales. El síntoma es un modelo que parece casi perfecto en validación (val
comparte el mismo sesgo) pero es frágil ante llamadas nuevas.

`DiagonalGaussian.fit` acepta ahora `group_ids` (un id de llamada por fila).
Cuando se provee, la varianza final es el MÁXIMO, dimensión a dimensión,
entre:
  - la varianza intra-segmento (MLE clásico, como antes), y
  - la varianza ENTRE llamadas (varianza de las medias por llamada),
    que es la que realmente refleja cuántas unidades independientes hay.
Tomar el máximo nunca angosta la gaussiana por debajo de lo que la variación
entre llamadas ya sugiere — es una corrección conservadora, no una
regularización arbitraria.
"""
from __future__ import annotations

import numpy as np

MIN_VAR = 1e-6  # piso de varianza para evitar log(0) / división por cero
MIN_GROUPS_FOR_SHRINKAGE = 3  # con menos llamadas independientes que esto, la
                                # varianza entre-llamadas es demasiado ruidosa
                                # para ser útil; se usa solo el MLE clásico.


def _between_group_variance(X: np.ndarray, group_ids: np.ndarray) -> np.ndarray | None:
    """Varianza de las medias por grupo (llamada). None si no hay grupos
    suficientes para que la estimación tenga sentido."""
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < MIN_GROUPS_FOR_SHRINKAGE:
        return None
    group_means = np.array([X[group_ids == g].mean(axis=0) for g in unique_groups])
    if len(unique_groups) < 2:
        return None
    return group_means.var(axis=0, ddof=1)


class DiagonalGaussian:
    """Gaussiana con covarianza diagonal para un bloque de features bajo una
    hipótesis (H0 o H1)."""

    def __init__(self, mean: np.ndarray | None = None, var: np.ndarray | None = None):
        self.mean = mean
        self.var = var

    def fit(self, X: np.ndarray, group_ids: np.ndarray | None = None) -> "DiagonalGaussian":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        var_segment = X.var(axis=0)

        var_final = var_segment
        if group_ids is not None:
            group_ids = np.asarray(group_ids)
            var_between = _between_group_variance(X, group_ids)
            if var_between is not None:
                # nunca angostar por debajo de lo que sugiere la variación
                # real entre llamadas independientes
                var_final = np.maximum(var_segment, var_between)

        self.var = np.maximum(var_final, MIN_VAR)
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
    atribución exacta por feature.

    `feature_mask` (opcional, un bool por feature) permite "apagar" en el
    LLR las features que resultaron poco discriminantes en entrenamiento
    (ver `train/fit_densities.py::select_feature_mask`), sin cambiar la
    dimensionalidad del vector de entrada: las gaussianas se siguen
    ajustando sobre todas las dimensiones (para que la atribución exacta
    siga siendo posible de inspeccionar), pero una feature enmascarada
    aporta 0 al LLR total en vez de sumar ruido de estimación con pocos
    datos."""

    def __init__(self, feature_names: list[str], feature_mask: np.ndarray | None = None):
        self.feature_names = list(feature_names)
        self.h0 = DiagonalGaussian()
        self.h1 = DiagonalGaussian()
        self.feature_mask = (
            np.asarray(feature_mask, dtype=bool)
            if feature_mask is not None
            else np.ones(len(self.feature_names), dtype=bool)
        )

    def fit(self, X_h0: np.ndarray, X_h1: np.ndarray,
             group_ids_h0: np.ndarray | None = None,
             group_ids_h1: np.ndarray | None = None) -> "BlockDensityModel":
        self.h0.fit(X_h0, group_ids=group_ids_h0)
        self.h1.fit(X_h1, group_ids=group_ids_h1)
        return self

    def llr(self, x: np.ndarray) -> float:
        terms = self.h1.log_prob_terms(x) - self.h0.log_prob_terms(x)
        return float((terms * self.feature_mask).sum())

    def llr_attribution(self, x: np.ndarray) -> dict:
        """LLR total + contribución exacta de cada feature (h1_term - h0_term).
        Las features enmascaradas se reportan con contribución 0.0."""
        t1 = self.h1.log_prob_terms(x)
        t0 = self.h0.log_prob_terms(x)
        per_feature = (t1 - t0) * self.feature_mask
        return {
            "llr_total": float(per_feature.sum()),
            "per_feature": {name: float(v) for name, v in zip(self.feature_names, per_feature)},
        }

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "feature_mask": self.feature_mask.tolist(),
            "h0": self.h0.to_dict(),
            "h1": self.h1.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockDensityModel":
        m = cls(d["feature_names"], feature_mask=d.get("feature_mask"))
        m.h0 = DiagonalGaussian.from_dict(d["h0"])
        m.h1 = DiagonalGaussian.from_dict(d["h1"])
        return m
