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


def plot_repeat_auc_stability(repeat_aucs: np.ndarray, out_path: str,
                               title="Estabilidad del AUC entre repeticiones de CV"):
    """Un punto por repetición de la validación cruzada repetida (cada
    repetición = una partición distinta en folds). Un modelo estable se ve
    como una nube apretada alrededor de la media; una nube muy dispersa
    avisa que el dataset todavía es demasiado chico para confiar en un
    solo número de AUC."""
    repeat_aucs = np.asarray(repeat_aucs, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(1, len(repeat_aucs) + 1)
    ax.scatter(x, repeat_aucs, color="#1f77b4", zorder=3)
    mean_auc = repeat_aucs.mean()
    std_auc = repeat_aucs.std()
    ax.axhline(mean_auc, color="black", linestyle="--", label=f"media={mean_auc:.3f}")
    ax.axhspan(mean_auc - std_auc, mean_auc + std_auc, color="gray", alpha=0.2,
               label=f"+/-1 std={std_auc:.3f}")
    ax.set_xlabel("Repetición de CV (partición distinta)")
    ax.set_ylabel("AUC (OOF de esa repetición)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _savefig(fig, out_path)


def plot_calibration_reliability(ece_report: dict, out_path: str,
                                  title="Diagrama de confiabilidad (calibración)"):
    """Confianza reportada (eje x) vs. tasa real de aciertos observada
    (eje y), por bins. La diagonal punteada es la calibración perfecta:
    si los puntos caen sistemáticamente por debajo/arriba de la diagonal,
    el modelo está sobre/subconfiado, respectivamente."""
    bins = ece_report.get("bins", [])
    xs, ys, sizes = [], [], []
    for b in bins:
        if b["count"] == 0:
            continue
        xs.append(b["mean_confidence"])
        ys.append(b["empirical_rate"])
        sizes.append(b["count"])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="calibración perfecta")
    if xs:
        sizes_arr = np.array(sizes, dtype=np.float64)
        ax.scatter(xs, ys, s=40 + 200 * sizes_arr / sizes_arr.max(), color="#d62728", alpha=0.8,
                   label="bins observados (tamaño = # llamadas)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confianza media reportada por el modelo")
    ax.set_ylabel("Tasa real de H1 (sintético) observada")
    ax.set_title(f"{title}\nECE={ece_report.get('ece', float('nan')):.3f}")
    ax.legend(loc="upper left", fontsize=7)
    _savefig(fig, out_path)


def generate_all_plots(out_dir, cm, val_y, val_scores, auc, eta,
                        all_llr_a, all_llr_b, all_y, train_mask, calibrator,
                        repeat_aucs=None, ece_report=None):
    """Genera y guarda todas las gráficas de diagnóstico en `out_dir`.
    Regresa la lista de rutas de los PNG generados. `repeat_aucs` y
    `ece_report` son opcionales (validación cruzada repetida / calibración);
    si no se proveen, simplemente no se generan esas dos gráficas."""
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

    if repeat_aucs is not None and len(repeat_aucs) > 1:
        p = out_dir / "cv_repeat_stability.png"
        plot_repeat_auc_stability(np.asarray(repeat_aucs), str(p))
        paths.append(p)

    if ece_report is not None:
        p = out_dir / "calibration_reliability.png"
        plot_calibration_reliability(ece_report, str(p))
        paths.append(p)

    return paths
