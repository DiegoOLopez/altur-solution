"""
Actualización online (sección 1.6): las estadísticas suficientes de una
gaussiana (mu, sigma^2 por dimensión) admiten actualización recursiva exacta
usando un prior conjugado Normal-Inverse-Gamma (NIG) por dimensión. Esto es
una actualización posterior bayesiana exacta, no un heurístico de
"reentrenar cada semana".

Para una dimensión con prior NIG(mu0, kappa0, alpha0, beta0) y n nuevas
observaciones con media x_bar y varianza muestral s2, el posterior es:

    kappa_n = kappa0 + n
    mu_n    = (kappa0*mu0 + n*x_bar) / kappa_n
    alpha_n = alpha0 + n/2
    beta_n  = beta0 + 0.5*n*s2 + (kappa0*n*(x_bar-mu0)^2) / (2*kappa_n)

El estimador puntual usado como (mu, sigma^2) actualizados de la
DiagonalGaussian es la media/moda del posterior:
    mu_hat     = mu_n
    sigma2_hat = beta_n / (alpha_n + 1)   (moda de Inverse-Gamma para sigma^2 dado el resto)
"""
from __future__ import annotations

import numpy as np

from model.density import DiagonalGaussian, MIN_VAR


class NIGState:
    """Estado NIG por dimensión, vectorizado sobre todas las dimensiones del
    bloque de una sola vez."""

    def __init__(self, mu0: np.ndarray, kappa0: np.ndarray, alpha0: np.ndarray, beta0: np.ndarray):
        self.mu = np.array(mu0, dtype=np.float64)
        self.kappa = np.array(kappa0, dtype=np.float64)
        self.alpha = np.array(alpha0, dtype=np.float64)
        self.beta = np.array(beta0, dtype=np.float64)

    @classmethod
    def from_gaussian(cls, gaussian: DiagonalGaussian, n_effective: float = 20.0) -> "NIGState":
        """Inicializa el prior NIG a partir de una DiagonalGaussian ya
        entrenada, tratándola como si viniera de `n_effective` observaciones
        (pseudo-cuenta que regula qué tanto puede mover el prior la nueva
        evidencia: mayor n_effective = prior más "terco")."""
        d = len(gaussian.mean)
        kappa0 = np.full(d, n_effective)
        alpha0 = np.full(d, n_effective / 2)
        beta0 = gaussian.var * (alpha0)  # E[sigma^2] = beta0/alpha0 = var original
        return cls(mu0=gaussian.mean.copy(), kappa0=kappa0, alpha0=alpha0, beta0=beta0)

    def update(self, X_new: np.ndarray) -> "NIGState":
        """Actualiza el estado con un batch de nuevas observaciones
        etiquetadas para esta clase (X_new: shape [n, d])."""
        X_new = np.asarray(X_new, dtype=np.float64)
        n = X_new.shape[0]
        if n == 0:
            return self
        x_bar = X_new.mean(axis=0)
        s2 = X_new.var(axis=0) if n > 1 else np.zeros(X_new.shape[1])

        kappa_n = self.kappa + n
        mu_n = (self.kappa * self.mu + n * x_bar) / kappa_n
        alpha_n = self.alpha + n / 2
        beta_n = (
            self.beta
            + 0.5 * n * s2
            + (self.kappa * n * (x_bar - self.mu) ** 2) / (2 * kappa_n)
        )

        self.mu, self.kappa, self.alpha, self.beta = mu_n, kappa_n, alpha_n, beta_n
        return self

    def point_estimate(self) -> DiagonalGaussian:
        sigma2_hat = np.maximum(self.beta / (self.alpha + 1), MIN_VAR)
        return DiagonalGaussian(mean=self.mu.copy(), var=sigma2_hat)

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "kappa": self.kappa.tolist(),
                "alpha": self.alpha.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NIGState":
        return cls(np.array(d["mu"]), np.array(d["kappa"]), np.array(d["alpha"]), np.array(d["beta"]))


def online_update_block(block_density_model, new_X_h0: np.ndarray, new_X_h1: np.ndarray, n_effective: float = 20.0):
    """Actualiza in-place un BlockDensityModel completo (H0 y H1) con nuevas
    llamadas etiquetadas, sin reentrenar desde cero."""
    if new_X_h0 is not None and len(new_X_h0) > 0:
        nig0 = NIGState.from_gaussian(block_density_model.h0, n_effective).update(new_X_h0)
        block_density_model.h0 = nig0.point_estimate()
    if new_X_h1 is not None and len(new_X_h1) > 0:
        nig1 = NIGState.from_gaussian(block_density_model.h1, n_effective).update(new_X_h1)
        block_density_model.h1 = nig1.point_estimate()
    return block_density_model
