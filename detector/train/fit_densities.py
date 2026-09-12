#!/usr/bin/env python3
"""
Entrena el detector completo y guarda el binario (.pkl) listo para
importarse en el API / dashboard.

Estructura de datos esperada:

    data/train/
        human/
            call001.wav
            call001.turns.json      (opcional; si falta se auto-diariza)
            call002.wav
            ...
        synthetic/
            call101.wav
            call101.turns.json
            ...

`call001.turns.json` tiene el mismo formato que el JSON de Altur:
    {"turns": [{"channel": 0|1, "start": t0, "end": t1}, ...]}

Uso:
    python -m train.fit_densities \
        --data_dir data/train \
        --out models/model.pkl \
        --prior_h1 0.3 \
        --val_split 0.2

El script:
  1. Recorre human/ y synthetic/, carga cada wav (canal 0 = caller).
  2. Extrae x_a por segmento (1s) y x_b por evento de turno.
  3. Separa train/val por LLAMADA (no por segmento) para no filtrar información.
  4. Ajusta DiagonalGaussian H0/H1 por bloque sobre el split de train.
  5. Calcula LLR_acustico_total y LLR_comportamental_total por llamada (train+val).
  6. Ajusta el stacking logístico sobre esos 2 scores + entrena el umbral eta
     (Youden) sobre el split de validación.
  7. Imprime un reporte (accuracy, AUC, matriz de confusión) y guarda el binario.
  8. Genera gráficas de diagnóstico (PNG) en --plots_dir: matriz de
     confusión, curva ROC, distribución del score, dispersión LLR por
     bloque y distribución del LLR de cada bloque por clase.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.acoustic import segment_features
from features.behavioral import behavioral_event_features
from preprocessing.diarization import diarize_stereo
from model.pipeline import VoiceAuthenticityDetector, load_wav_any, split_channels, SEGMENT_SECONDS
from model.calibration import choose_threshold_youden
from train.plots import generate_all_plots

LABELS = {"human": 0, "synthetic": 1}  # H0=0 (humano), H1=1 (sintético)


def find_calls(data_dir: Path):
    """Regresa lista de (wav_path, label:int, turns_path_or_None)."""
    calls = []
    for label_name, label in LABELS.items():
        folder = data_dir / label_name
        if not folder.exists():
            continue
        for wav_path in sorted(folder.glob("*.wav")):
            turns_path = wav_path.with_suffix("").with_suffix(".turns.json")
            calls.append((wav_path, label, turns_path if turns_path.exists() else None))
    return calls


def load_turns(turns_path: Path | None, signal, sr) -> list[dict]:
    if turns_path is not None:
        with open(turns_path) as f:
            return json.load(f)["turns"]
    if signal.ndim == 2 and signal.shape[1] >= 2:
        return diarize_stereo(signal, sr)
    return []


def extract_call_features(wav_path: Path, turns_path: Path | None):
    signal, sr = load_wav_any(str(wav_path))
    caller, agent = split_channels(signal)
    turns = load_turns(turns_path, signal, sr)

    seg_len = int(SEGMENT_SECONDS * sr)
    n_segments = max(len(caller) // seg_len, 0)
    X_a = np.array([segment_features(caller[i * seg_len:(i + 1) * seg_len], sr)
                     for i in range(n_segments)]) if n_segments else np.empty((0, 1))

    behavioral = behavioral_event_features(turns) if turns else []
    X_b = np.array([x for _, x in behavioral]) if behavioral else np.empty((0, 1))

    return X_a, X_b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/train", help="carpeta con human/ y synthetic/")
    ap.add_argument("--out", default="models/model.pkl", help="ruta de salida del binario")
    ap.add_argument("--prior_h1", type=float, default=0.3, help="prior p(voz sintética)")
    ap.add_argument("--val_split", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots_dir", default="reports", help="carpeta de salida de las gráficas PNG (usa '' para desactivarlas)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    calls = find_calls(data_dir)
    if len(calls) < 4:
        print(f"[!] Muy pocas llamadas encontradas en {data_dir} ({len(calls)}). "
              f"Se necesitan al menos ~2 por clase. Revisa data_dir/human y data_dir/synthetic.")
        sys.exit(1)

    labels = [c[1] for c in calls]
    idx_train, idx_val = train_test_split(
        range(len(calls)), test_size=args.val_split, random_state=args.seed, stratify=labels
    )

    print(f"[i] {len(calls)} llamadas encontradas | train={len(idx_train)} val={len(idx_val)}")
    print("[i] Extrayendo features por llamada (puede tardar un poco)...")

    per_call = {}
    for i, (wav_path, label, turns_path) in enumerate(calls):
        X_a, X_b = extract_call_features(wav_path, turns_path)
        per_call[i] = {"X_a": X_a, "X_b": X_b, "label": label, "path": str(wav_path)}
        print(f"    [{i+1}/{len(calls)}] {wav_path.name}: "
              f"{X_a.shape[0]} segmentos acústicos, {X_b.shape[0]} eventos comportamentales")

    detector = VoiceAuthenticityDetector()
    detector.prior_h1 = args.prior_h1

    # ---- 1) ajustar densidades por bloque sobre TRAIN, agrupando todos los
    # segmentos/eventos de todas las llamadas de cada clase ----
    train_h0_a, train_h1_a = [], []
    train_h0_b, train_h1_b = [], []
    for i in idx_train:
        d = per_call[i]
        target_a = train_h1_a if d["label"] == 1 else train_h0_a
        target_b = train_h1_b if d["label"] == 1 else train_h0_b
        if d["X_a"].shape[1] > 1:
            target_a.extend(d["X_a"])
        if d["X_b"].shape[1] > 1:
            target_b.extend(d["X_b"])

    def stack_or_zeros(rows, n_dims):
        return np.array(rows) if rows else np.zeros((2, n_dims))

    n_dim_a = next((d["X_a"].shape[1] for d in per_call.values() if d["X_a"].shape[1] > 1), len(detector.acoustic_block.feature_names))
    n_dim_b = next((d["X_b"].shape[1] for d in per_call.values() if d["X_b"].shape[1] > 1), len(detector.behavioral_block.feature_names))

    detector.acoustic_block.fit(stack_or_zeros(train_h0_a, n_dim_a), stack_or_zeros(train_h1_a, n_dim_a))
    detector.behavioral_block.fit(stack_or_zeros(train_h0_b, n_dim_b), stack_or_zeros(train_h1_b, n_dim_b))
    print(f"[i] Densidades ajustadas: bloque acústico n_h0={len(train_h0_a)} n_h1={len(train_h1_a)} | "
          f"bloque comportamental n_h0={len(train_h0_b)} n_h1={len(train_h1_b)}")

    # ---- 2) calcular LLR_total por bloque, por llamada (train+val), para el stacking ----
    def call_llrs(d):
        llr_a = sum(detector.acoustic_block.llr(x) for x in d["X_a"]) if d["X_a"].shape[1] > 1 else 0.0
        llr_b = sum(detector.behavioral_block.llr(x) for x in d["X_b"]) if d["X_b"].shape[1] > 1 else 0.0
        return llr_a, llr_b

    all_llr_a, all_llr_b, all_y = [], [], []
    for i in range(len(calls)):
        d = per_call[i]
        llr_a, llr_b = call_llrs(d)
        all_llr_a.append(llr_a)
        all_llr_b.append(llr_b)
        all_y.append(d["label"])
    all_llr_a, all_llr_b, all_y = map(np.array, (all_llr_a, all_llr_b, all_y))

    train_mask = np.zeros(len(calls), dtype=bool)
    train_mask[list(idx_train)] = True

    # ---- 3) ajustar stacking logístico sobre TRAIN ----
    detector.calibrator.fit(all_llr_a[train_mask], all_llr_b[train_mask], all_y[train_mask])
    print(f"[i] Stacking logístico ajustado: {detector.calibrator.weights}")

    # ---- 4) elegir eta (umbral) sobre VAL usando el score LLR total ----
    val_scores = all_llr_a[~train_mask] + all_llr_b[~train_mask]
    val_y = all_y[~train_mask]
    detector.eta = choose_threshold_youden(val_scores, val_y)

    # ---- 5) reporte sobre VAL ----
    val_pred = (val_scores > detector.eta).astype(int)
    try:
        auc = roc_auc_score(val_y, val_scores)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(val_y, val_pred)
    cm = confusion_matrix(val_y, val_pred, labels=[0, 1])

    detector.train_report = {
        "n_calls_total": len(calls),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "eta": detector.eta,
        "val_auc": float(auc),
        "val_accuracy": float(acc),
        "val_confusion_matrix": cm.tolist(),
        "stacking_weights": detector.calibrator.weights,
    }

    print("\n===== REPORTE DE VALIDACIÓN =====")
    print(f"  eta (umbral score LLR)  : {detector.eta:.3f}")
    print(f"  AUC                     : {auc:.3f}")
    print(f"  Accuracy                : {acc:.3f}")
    print(f"  Matriz de confusión     :\n{cm}  (filas=real[human,synth], cols=pred)")
    print(f"  Pesos stacking          : {detector.calibrator.weights}")

    if args.plots_dir:
        plot_paths = generate_all_plots(
            args.plots_dir, cm, val_y, val_scores, auc, detector.eta,
            all_llr_a, all_llr_b, all_y, train_mask, detector.calibrator,
        )
        print("\n[i] Gráficas de diagnóstico guardadas en:")
        for p in plot_paths:
            print(f"    - {p}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(out_path))
    print(f"\n[✔] Modelo guardado en: {out_path}")


if __name__ == "__main__":
    main()
