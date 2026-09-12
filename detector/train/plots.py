"""
Gráficas de diagnóstico del entrenamiento (guardadas como PNG).

Todas las funciones reciben datos ya calculados por `train/fit_densities.py`
(nunca recalculan nada) y solo se encargan de dibujar y guardar la figura.
Usan matplotlib con backend "Agg" (sin ventana) para poder correr en
servidores/CI sin display.

Gráficas incluidas:
  - Matriz de confusión (val)
  - Curva ROC (val) con el punto operativo elegido (eta)
  - Distribución del score total por clase (val), con la línea de eta
  - Dispersión LLR_acústico vs LLR_comportamental por llamada (train+val),
    con la frontera de decisión del stacking logístico
  - Distribución del LLR de cada bloque por clase (train+val)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

CLASS_NAMES = ["humano", "sintético"]


def _savefig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str, title="Matriz de confusión (validación)"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _savefig(fig, out_path)


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, auc: float, eta: float,
                    out_path: str, title="Curva ROC (validación)"):
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "No hay suficientes clases\nen validación para ROC",
                 ha="center", va="center")
    else:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, color="tab:blue", label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
        # punto operativo real: el más cercano a eta entre los thresholds de la curva
        idx = int(np.argmin(np.abs(thresholds - eta)))
        ax.scatter([fpr[idx]], [tpr[idx]], color="tab:red", zorder=5,
                    label=f"Umbral eta = {eta:.2f}")
        ax.legend(loc="lower right")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _savefig(fig, out_path)


def plot_score_distribution(scores: np.ndarray, y: np.ndarray, eta: float, out_path: str,
                              title="Distribución del score(t) total (validación)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {0: "tab:green", 1: "tab:red"}
    for label in (0, 1):
        vals = scores[y == label]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=min(10, max(3, len(vals))), alpha=0.6,
                 color=colors[label], label=CLASS_NAMES[label])
    ax.axvline(eta, color="black", linestyle="--", label=f"eta = {eta:.2f}")
    ax.set_xlabel("score_total = LLR_acústico + LLR_comportamental")
    ax.set_ylabel("Número de llamadas")
    ax.set_title(title)
    ax.legend()
    _savefig(fig, out_path)


def plot_llr_scatter(llr_a: np.ndarray, llr_b: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                      calibrator, out_path: str,
                      title="LLR acústico vs comportamental, por llamada"):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {0: "tab:green", 1: "tab:red"}
    markers = {True: "o", False: "^"}
    for label in (0, 1):
        for is_train in (True, False):
            mask = (y == label) & (train_mask == is_train)
            if not mask.any():
                continue
            ax.scatter(llr_a[mask], llr_b[mask], c=colors[label], marker=markers[is_train],
                        edgecolor="black", linewidth=0.4, alpha=0.85,
                        label=f"{CLASS_NAMES[label]} ({'train' if is_train else 'val'})")

    # frontera de decisión del stacking logístico: w_a*a + w_b*b + bias = 0
    w = calibrator.weights
    xs = np.linspace(min(llr_a.min(), 0) - 1, max(llr_a.max(), 0) + 1, 50)
    mean_, std_ = calibrator.mean_, calibrator.std_
    # las variables del stacking están estandarizadas; se despeja b en escala cruda
    try:
        a_std = (xs - mean_[0]) / std_[0]
        b_std = -(w["w_acoustic"] * a_std + w["bias"]) / max(w["w_behavioral"], 1e-9)
        b_raw = b_std * std_[1] + mean_[1]
        ax.plot(xs, b_raw, color="black", linestyle="--", linewidth=1,
                 label="Frontera del stacking (p=0.5)")
    except Exception:
        pass

    ax.set_xlabel("LLR_acústico_total")
    ax.set_ylabel("LLR_comportamental_total")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _savefig(fig, out_path)


def plot_block_llr_distributions(llr_a: np.ndarray, llr_b: np.ndarray, y: np.ndarray, out_path: str,
                                   title="Distribución del LLR por bloque y clase"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {0: "tab:green", 1: "tab:red"}
    for ax, values, name in zip(axes, (llr_a, llr_b), ("Acústico", "Comportamental")):
        for label in (0, 1):
            vals = values[y == label]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=min(10, max(3, len(vals))), alpha=0.6,
                     color=colors[label], label=CLASS_NAMES[label])
        ax.set_title(f"LLR {name}")
        ax.set_xlabel("LLR_total de la llamada")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    _savefig(fig, out_path)


def generate_all_plots(out_dir, cm, val_y, val_scores, auc, eta,
                        all_llr_a, all_llr_b, all_y, train_mask, calibrator):
    """Genera y guarda todas las gráficas de diagnóstico en `out_dir`.
    Regresa la lista de rutas de los PNG generados."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    p = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, str(p))
    paths.append(p)

    p = out_dir / "roc_curve.png"
    plot_roc_curve(val_y, val_scores, auc, eta, str(p))
    paths.append(p)

    p = out_dir / "score_distribution.png"
    plot_score_distribution(val_scores, val_y, eta, str(p))
    paths.append(p)

    p = out_dir / "llr_scatter.png"
    plot_llr_scatter(all_llr_a, all_llr_b, all_y, train_mask, calibrator, str(p))
    paths.append(p)

    p = out_dir / "llr_block_distributions.png"
    plot_block_llr_distributions(all_llr_a, all_llr_b, all_y, str(p))
    paths.append(p)

    return paths
