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
        --n_folds 5 \
        --n_repeats 20 \
        --n_augments 4

--- Qué hace distinto de una versión "ingenua" (y por qué, con datasets
    chicos de un par de cientos/miles de llamadas, importa) ---

1. Compuerta de energía: descarta segmentos de silencio antes de ajustar
   las gaussianas (`features.acoustic.is_active_segment`). Sin esto, ambas
   clases acumulan el mismo vector ~cero en los segmentos silenciosos, lo
   que comprime artificialmente sigma en varias dimensiones. Se usa la
   MISMA compuerta en inferencia (`model/pipeline.py`), para que el modelo
   vea la misma distribución de segmentos en entrenamiento y en producción.

2. Shrinkage de varianza por llamada: los segmentos/eventos de una misma
   llamada no son i.i.d. (mismo hablante). `DiagonalGaussian.fit` recibe un
   id de llamada por fila y usa el máximo entre la varianza intra-segmento
   clásica y la varianza ENTRE llamadas, para no subestimar sigma solo
   porque hay muchos segmentos por pocas llamadas reales.

3. Selección de features por AUC a nivel de LLAMADA: antes de ajustar cada
   bloque, se mide qué tan discriminante es cada dimensión usando el
   promedio por llamada (no por segmento, para no inflar la señal con
   pseudo-repetición) y se apagan del LLR (vía `feature_mask`) las que no
   superan un margen mínimo. Esto reduce el ruido de estimación que
   introducen features poco informativas cuando hay pocos datos.

4. Aumento de datos (augmentación) — SOLO para el split de TRAIN de cada
   fold, nunca para validación. Cada llamada real se acompaña de N
   variantes con condiciones de canal/entorno distintas (ruido, ancho de
   banda telefónico, códec, pérdida de paquetes, reverberación leve,
   ganancia) generadas por `preprocessing/augmentation.py`. No se inventan
   llamadas nuevas (mismo hablante, mismo contenido); solo se diversifican
   las condiciones de grabación bajo las que el modelo ve esa evidencia.
   Las variantes de una llamada comparten su `group_id` (índice de la
   llamada original) para el shrinkage de varianza por llamada — se tratan
   como más evidencia de la MISMA llamada, no como llamadas independientes
   nuevas (ver docstring de `preprocessing/augmentation.py`).

5. Validación cruzada REPETIDA y estratificada por LLAMADA (k-fold, muchas
   veces con particiones distintas): en cada fold se ajustan densidades +
   máscara de features + stacking sobre el fold de train (incluyendo sus
   variantes aumentadas), y se evalúa el fold de val (SIEMPRE con audio
   real, nunca aumentado). Los scores OOF de todas las repeticiones se
   promedian por llamada; además se reporta la media ± std del AUC ENTRE
   repeticiones, que es la métrica de estabilidad que de verdad importa
   con un dataset de tamaño moderado: si esa std es alta, el modelo (o el
   dataset) todavía no es confiable, sin importar qué tan bueno se vea un
   único split.

6. El modelo que se GUARDA para producción se reajusta al final sobre el
   100% de las llamadas (con la misma máscara de features y el mismo tipo
   de shrinkage, más augmentación), pero el umbral eta y el hiperparámetro
   de regularización del stacking (`C`) se eligen sobre el promedio de la
   validación cruzada repetida — nunca sobre datos que el modelo final ya
   vio en su forma real — para no filtrar información al umbral de
   decisión ni al hiperparámetro.

7. Métricas de calibración (Brier, ECE) e intervalo de confianza por
   bootstrap del AUC: el producto final no es solo un veredicto binario,
   es una confianza en [0,1] — estas métricas dicen qué tan honesta es esa
   confianza y cuánta incertidumbre hay en el AUC reportado dado el tamaño
   del dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.acoustic import segment_features, is_active_segment, ACOUSTIC_FEATURE_NAMES
from features.behavioral import behavioral_event_features, BEHAVIORAL_FEATURE_NAMES
from preprocessing.diarization import diarize_stereo
from preprocessing.augmentation import generate_augmented_variants
from model.pipeline import VoiceAuthenticityDetector, load_wav_any, split_channels, SEGMENT_SECONDS
from model.calibration import choose_threshold_youden, StackingCalibrator
from model.density import BlockDensityModel
from train.plots import generate_all_plots
from train.metrics import brier_score, expected_calibration_error, bootstrap_auc_ci

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


def acoustic_features_from_caller(caller: np.ndarray, sr: int):
    """Segmenta una señal (canal del caller, ya sea original o una
    variante aumentada) en bloques de SEGMENT_SECONDS, descarta silencio y
    extrae `segment_features` de los segmentos activos. Se usa igual para
    audio real y para variantes aumentadas, para no duplicar lógica."""
    seg_len = int(SEGMENT_SECONDS * sr)
    n_segments = max(len(caller) // seg_len, 0)
    ref_peak = float(np.abs(caller).max()) if caller.size else 1e-9

    active_rows = []
    n_discarded = 0
    for i in range(n_segments):
        seg = caller[i * seg_len:(i + 1) * seg_len]
        if is_active_segment(seg, ref_peak):
            active_rows.append(segment_features(seg, sr))
        else:
            n_discarded += 1
    X = np.array(active_rows) if active_rows else np.empty((0, 1))
    return X, n_discarded


def extract_call_features(wav_path: Path, turns_path: Path | None, n_augments: int, augment_rng):
    """Extrae X_a (segmentos acústicos ACTIVOS del audio real), X_b
    (eventos comportamentales) y, si `n_augments` > 0, una lista de X_a
    adicionales — una por variante aumentada de la misma llamada (mismo
    turns.json, mismo canal del agente sin tocar; solo cambia la condición
    de canal/entorno del audio del caller). Las variantes se generan aquí,
    UNA sola vez por llamada, y se reutilizan en todas las repeticiones /
    folds de la validación cruzada en los que esa llamada caiga en train."""
    signal, sr = load_wav_any(str(wav_path))
    caller, agent = split_channels(signal)
    turns = load_turns(turns_path, signal, sr)

    X_a, n_discarded_silence = acoustic_features_from_caller(caller, sr)

    behavioral = behavioral_event_features(turns) if turns else []
    X_b = np.array([x for _, x in behavioral]) if behavioral else np.empty((0, 1))

    X_a_aug = []
    if n_augments > 0:
        for _tag, caller_variant in generate_augmented_variants(caller, sr, augment_rng, n_augments):
            X_a_var, _ = acoustic_features_from_caller(caller_variant, sr)
            X_a_aug.append(X_a_var)

    return X_a, X_b, X_a_aug, n_discarded_silence


def select_feature_mask(X_call_level: np.ndarray, y_call: np.ndarray,
                         margin: float, min_keep: int) -> np.ndarray:
    """True para las features que se conservan activas en el LLR.

    Se mide discriminabilidad univariada por AUC sobre el promedio POR
    LLAMADA de cada feature (no por segmento: promediar por llamada evita
    tratar segmentos correlacionados de la misma llamada como evidencia
    independiente al elegir qué features son útiles). Una feature con
    AUC ~ 0.5 no separa las clases; se apaga para no sumar ruido de
    estimación al ajustar su gaussiana con pocos datos. Si muy pocas
    superan `margin`, se conservan al menos `min_keep` (las de mayor
    discriminabilidad), para no dejar el bloque vacío."""
    n_dims = X_call_level.shape[1]
    if len(np.unique(y_call)) < 2 or X_call_level.shape[0] < 4:
        return np.ones(n_dims, dtype=bool)

    disc = np.zeros(n_dims)
    for j in range(n_dims):
        col = X_call_level[:, j]
        if np.std(col) < 1e-9:
            continue
        try:
            auc = roc_auc_score(y_call, col)
        except ValueError:
            auc = 0.5
        disc[j] = abs(auc - 0.5)

    mask = disc >= margin
    if mask.sum() < min_keep:
        top = np.argsort(disc)[::-1][:min_keep]
        mask = np.zeros(n_dims, dtype=bool)
        mask[top] = True
    return mask


def call_level_matrix(per_call: dict, indices, key: str, n_dims: int) -> np.ndarray:
    """Promedio por llamada de un bloque de features (para selección de
    features y para no inflar N con pseudo-repetición intra-llamada).
    Usa SOLO el audio real (nunca variantes aumentadas): la selección de
    features debe reflejar qué discrimina en llamadas reales, no en
    condiciones de canal sintéticas."""
    rows = []
    for i in indices:
        X = per_call[i][key]
        if X.ndim == 2 and X.shape[1] > 1 and X.shape[0] > 0:
            rows.append(X.mean(axis=0))
        else:
            rows.append(np.zeros(n_dims))
    return np.array(rows)


def gather_train_rows(per_call: dict, indices, key: str, n_dim: int,
                       include_augmented: bool = False):
    """Junta filas + id de llamada de todas las llamadas de `indices`,
    separado por clase (H0/H1), para ajustar un BlockDensityModel con
    shrinkage de varianza por llamada.

    Si `include_augmented` es True y `key == "X_a"`, también se agregan
    las variantes aumentadas de cada llamada, todas con el MISMO id de
    grupo que la llamada original (`i`) — así el shrinkage de varianza
    entre-llamadas sigue viendo el número real de llamadas independientes,
    no llamadas-más-sus-clones como si fueran independientes."""
    rows_h0, rows_h1 = [], []
    groups_h0, groups_h1 = [], []
    for i in indices:
        d = per_call[i]
        mats = [d[key]]
        if include_augmented and key == "X_a":
            mats = mats + d.get("X_a_aug", [])
        target_rows = rows_h1 if d["label"] == 1 else rows_h0
        target_groups = groups_h1 if d["label"] == 1 else groups_h0
        for X in mats:
            if X.ndim != 2 or X.shape[1] <= 1 or X.shape[0] == 0:
                continue
            target_rows.extend(X)
            target_groups.extend([i] * X.shape[0])

    def _finalize(rows, groups):
        if not rows:
            return np.zeros((2, n_dim)), np.array([0, 1])
        return np.array(rows), np.array(groups)

    X_h0, g_h0 = _finalize(rows_h0, groups_h0)
    X_h1, g_h1 = _finalize(rows_h1, groups_h1)
    return X_h0, g_h0, X_h1, g_h1


def fit_block(per_call: dict, indices, key: str, feature_names: list[str], n_dim: int,
              feature_mask: np.ndarray | None = None, include_augmented: bool = False) -> BlockDensityModel:
    X_h0, g_h0, X_h1, g_h1 = gather_train_rows(per_call, indices, key, n_dim, include_augmented=include_augmented)
    block = BlockDensityModel(feature_names, feature_mask=feature_mask)
    block.fit(X_h0, X_h1, group_ids_h0=g_h0, group_ids_h1=g_h1)
    return block


def call_llrs(d: dict, acoustic_block: BlockDensityModel, behavioral_block: BlockDensityModel,
              X_a_override: np.ndarray | None = None):
    """LLR acústico + comportamental de una llamada. `X_a_override` permite
    calcular el LLR acústico de una VARIANTE aumentada en vez del audio
    real, reutilizando el X_b real (la augmentación de canal no cambia los
    tiempos de turno)."""
    X_a = X_a_override if X_a_override is not None else d["X_a"]
    llr_a = sum(acoustic_block.llr(x) for x in X_a) if X_a.ndim == 2 and X_a.shape[1] > 1 else 0.0
    llr_b = sum(behavioral_block.llr(x) for x in d["X_b"]) if d["X_b"].ndim == 2 and d["X_b"].shape[1] > 1 else 0.0
    return llr_a, llr_b


def training_llr_pairs(per_call: dict, indices, acoustic_block: BlockDensityModel,
                        behavioral_block: BlockDensityModel, include_augmented: bool):
    """LLR (acústico, comportamental) por llamada de TRAIN para ajustar el
    stacking — si `include_augmented`, cada variante aumentada de una
    llamada aporta también su propio par (llr_a_variante, llr_b_real),
    como ejemplos adicionales legítimos (misma llamada, distinta condición
    de canal), no como llamadas nuevas inventadas."""
    llr_a_list, llr_b_list, y_list = [], [], []
    for i in indices:
        d = per_call[i]
        la, lb = call_llrs(d, acoustic_block, behavioral_block)
        llr_a_list.append(la)
        llr_b_list.append(lb)
        y_list.append(d["label"])
        if include_augmented:
            for X_a_var in d.get("X_a_aug", []):
                la_v, _ = call_llrs(d, acoustic_block, behavioral_block, X_a_override=X_a_var)
                llr_a_list.append(la_v)
                llr_b_list.append(lb)
                y_list.append(d["label"])
    return np.array(llr_a_list), np.array(llr_b_list), np.array(y_list)


def run_repeated_cv(per_call: dict, all_y: np.ndarray, n_folds: int, n_repeats: int,
                     seed_base: int, feature_auc_margin: float, min_features_acoustic: int,
                     min_features_behavioral: int, n_dim_a: int, n_dim_b: int,
                     use_augment: bool, stacking_C: float):
    """Corre `n_repeats` particiones k-fold distintas (semillas distintas),
    cada una cubriendo el 100% de las llamadas exactamente una vez (OOF).
    Regresa, por llamada: matriz (n_repeats x n_calls) de LLR-score y de
    probabilidad calibrada OOF, más estadísticas de estabilidad de
    features y AUC por repetición."""
    n_calls = len(all_y)
    all_indices = np.arange(n_calls)
    labels = all_y.tolist()

    oof_score_reps = np.zeros((n_repeats, n_calls))
    oof_prob_reps = np.zeros((n_repeats, n_calls))
    per_repeat_auc = []
    feature_keep_counts_a = np.zeros(n_dim_a)
    feature_keep_counts_b = np.zeros(n_dim_b)
    total_folds = 0
    last_fold_reports = []

    for rep in range(n_repeats):
        seed_rep = seed_base + rep
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_rep)
        oof_llr_a = np.zeros(n_calls)
        oof_llr_b = np.zeros(n_calls)
        oof_prob = np.zeros(n_calls)
        fold_reports = []

        for fold_i, (idx_train, idx_val) in enumerate(skf.split(all_indices, labels)):
            Xa_call = call_level_matrix(per_call, idx_train, "X_a", n_dim_a)
            Xb_call = call_level_matrix(per_call, idx_train, "X_b", n_dim_b)
            y_train = all_y[idx_train]
            mask_a = select_feature_mask(Xa_call, y_train, feature_auc_margin, min_features_acoustic)
            mask_b = select_feature_mask(Xb_call, y_train, feature_auc_margin, min_features_behavioral)
            feature_keep_counts_a += mask_a
            feature_keep_counts_b += mask_b
            total_folds += 1

            acoustic_block = fit_block(per_call, idx_train, "X_a", ACOUSTIC_FEATURE_NAMES, n_dim_a,
                                        mask_a, include_augmented=use_augment)
            behavioral_block = fit_block(per_call, idx_train, "X_b", BEHAVIORAL_FEATURE_NAMES, n_dim_b,
                                          mask_b, include_augmented=False)

            train_llr_a, train_llr_b, train_y = training_llr_pairs(
                per_call, idx_train, acoustic_block, behavioral_block, include_augmented=use_augment)
            calibrator = StackingCalibrator(C=stacking_C)
            calibrator.fit(train_llr_a, train_llr_b, train_y)

            for i in idx_val:
                la, lb = call_llrs(per_call[i], acoustic_block, behavioral_block)
                oof_llr_a[i] = la
                oof_llr_b[i] = lb
                oof_prob[i] = calibrator.predict_proba(la, lb)

            val_score_fold = oof_llr_a[idx_val] + oof_llr_b[idx_val]
            val_y_fold = all_y[idx_val]
            try:
                auc_fold = roc_auc_score(val_y_fold, val_score_fold) if len(np.unique(val_y_fold)) > 1 else float("nan")
            except ValueError:
                auc_fold = float("nan")
            fold_reports.append({
                "fold": fold_i, "n_train": len(idx_train), "n_val": len(idx_val),
                "n_features_acoustic_kept": int(mask_a.sum()), "n_features_behavioral_kept": int(mask_b.sum()),
                "val_auc": float(auc_fold),
            })

        oof_score_reps[rep] = oof_llr_a + oof_llr_b
        oof_prob_reps[rep] = oof_prob
        try:
            auc_rep = roc_auc_score(all_y, oof_score_reps[rep])
        except ValueError:
            auc_rep = float("nan")
        per_repeat_auc.append(auc_rep)
        last_fold_reports = fold_reports
        print(f"    [repeat {rep + 1}/{n_repeats}] seed={seed_rep} AUC(OOF esta repetición)={auc_rep:.3f}")

    return {
        "oof_score_reps": oof_score_reps,
        "oof_prob_reps": oof_prob_reps,
        "per_repeat_auc": np.array(per_repeat_auc, dtype=np.float64),
        "feature_keep_freq_a": feature_keep_counts_a / max(total_folds, 1),
        "feature_keep_freq_b": feature_keep_counts_b / max(total_folds, 1),
        "total_folds": total_folds,
        "last_fold_reports": last_fold_reports,
    }


def per_call_llrs(per_call: dict, indices, acoustic_block: BlockDensityModel, behavioral_block: BlockDensityModel):
    llr_a = np.zeros(len(indices))
    llr_b = np.zeros(len(indices))
    for pos, i in enumerate(indices):
        la, lb = call_llrs(per_call[i], acoustic_block, behavioral_block)
        llr_a[pos] = la
        llr_b[pos] = lb
    return llr_a, llr_b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/train", help="carpeta con human/ y synthetic/")
    ap.add_argument("--out", default="models/model.pkl", help="ruta de salida del binario")
    ap.add_argument("--prior_h1", type=float, default=0.3, help="prior p(voz sintética)")
    ap.add_argument("--n_folds", type=int, default=5, help="folds de la validación cruzada por llamada, en cada repetición")
    ap.add_argument("--n_repeats", type=int, default=20,
                     help="repeticiones de la validación cruzada con particiones distintas (estabilidad, no 'épocas')")
    ap.add_argument("--seed", type=int, default=42, help="semilla base; cada repetición usa seed+rep")
    ap.add_argument("--feature_auc_margin", type=float, default=0.03,
                     help="|AUC-0.5| mínimo para que una feature se mantenga activa en el LLR")
    ap.add_argument("--min_features_acoustic", type=int, default=8,
                     help="mínimo de features acústicas a conservar aunque no superen el margen")
    ap.add_argument("--min_features_behavioral", type=int, default=2,
                     help="mínimo de features comportamentales a conservar aunque no superen el margen")
    ap.add_argument("--n_augments", type=int, default=4,
                     help="variantes aumentadas por llamada (condiciones de canal/entorno). 0 desactiva la augmentación")
    ap.add_argument("--augment_seed", type=int, default=None,
                     help="semilla de la augmentación (default: --seed). Se genera UNA vez por llamada, no por fold/repeat")
    ap.add_argument("--stacking_C_grid", default="0.5",
                     help="lista separada por comas de valores de C a probar para el stacking logístico; "
                          "se elige el que dé mayor AUC medio en la CV repetida (ej. '0.1,0.3,0.5,1,2')")
    ap.add_argument("--ece_bins", type=int, default=10, help="bins para el Expected Calibration Error")
    ap.add_argument("--n_bootstrap", type=int, default=2000, help="remuestras para el IC del AUC final")
    ap.add_argument("--plots_dir", default="reports", help="carpeta de salida de las gráficas PNG (usa '' para desactivarlas)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    calls = find_calls(data_dir)
    if len(calls) < 4:
        print(f"[!] Muy pocas llamadas encontradas en {data_dir} ({len(calls)}). "
              f"Se necesitan al menos ~2 por clase. Revisa data_dir/human y data_dir/synthetic.")
        sys.exit(1)

    n_folds = max(2, min(args.n_folds, min(sum(1 for c in calls if c[1] == 0),
                                            sum(1 for c in calls if c[1] == 1))))
    if n_folds != args.n_folds:
        print(f"[i] --n_folds ajustado a {n_folds} (no hay suficientes llamadas por clase para {args.n_folds} folds)")

    use_augment = args.n_augments > 0
    augment_seed = args.augment_seed if args.augment_seed is not None else args.seed

    labels = [c[1] for c in calls]
    print(f"[i] {len(calls)} llamadas encontradas | {n_folds}-fold CV x {args.n_repeats} repeticiones "
          f"(particiones distintas) | augmentación: "
          f"{'ON (' + str(args.n_augments) + ' variantes/llamada)' if use_augment else 'OFF'}")
    print("[i] Extrayendo features por llamada (puede tardar un poco; la augmentación se genera una sola vez aquí)...")

    per_call = {}
    total_discarded = 0
    total_aug_segments = 0
    for i, (wav_path, label, turns_path) in enumerate(calls):
        augment_rng = np.random.default_rng((augment_seed, i))  # determinista por llamada, independiente del split
        X_a, X_b, X_a_aug, n_discarded = extract_call_features(
            wav_path, turns_path, args.n_augments if use_augment else 0, augment_rng)
        total_discarded += n_discarded
        n_aug_segs = sum(x.shape[0] for x in X_a_aug if x.ndim == 2 and x.shape[1] > 1)
        total_aug_segments += n_aug_segs
        per_call[i] = {"X_a": X_a, "X_b": X_b, "X_a_aug": X_a_aug, "label": label, "path": str(wav_path)}
        print(f"    [{i+1}/{len(calls)}] {wav_path.name}: "
              f"{X_a.shape[0]} segmentos acústicos activos (+{n_discarded} silenciosos descartados), "
              f"{X_b.shape[0]} eventos comportamentales"
              + (f", +{len(X_a_aug)} variantes aumentadas ({n_aug_segs} segmentos extra)" if use_augment else ""))

    n_dim_a = next((d["X_a"].shape[1] for d in per_call.values() if d["X_a"].ndim == 2 and d["X_a"].shape[1] > 1),
                    len(ACOUSTIC_FEATURE_NAMES))
    n_dim_b = next((d["X_b"].shape[1] for d in per_call.values() if d["X_b"].ndim == 2 and d["X_b"].shape[1] > 1),
                    len(BEHAVIORAL_FEATURE_NAMES))
    all_y = np.array(labels)
    all_indices = np.arange(len(calls))

    print(f"[i] Total de segmentos silenciosos descartados en todo el dataset: {total_discarded}")
    if use_augment:
        print(f"[i] Total de segmentos acústicos adicionales por augmentación (solo se usan en TRAIN de cada fold): "
              f"{total_aug_segments}")

    # =========================================================
    # 1) VALIDACIÓN CRUZADA REPETIDA (k-fold estratificado por llamada,
    #    repetida con distintas particiones) -> grid opcional de C
    # =========================================================
    c_grid = [float(c.strip()) for c in args.stacking_C_grid.split(",") if c.strip()]
    best_c = c_grid[0]
    best_result = None
    best_mean_auc = -1.0
    grid_summary = {}

    for c in c_grid:
        if len(c_grid) > 1:
            print(f"\n[i] Probando C={c} para el stacking logístico (evaluado sobre la CV repetida completa)...")
        result = run_repeated_cv(
            per_call, all_y, n_folds, args.n_repeats, args.seed,
            args.feature_auc_margin, args.min_features_acoustic, args.min_features_behavioral,
            n_dim_a, n_dim_b, use_augment, stacking_C=c,
        )
        mean_score = result["oof_score_reps"].mean(axis=0)
        try:
            auc_mean_c = roc_auc_score(all_y, mean_score)
        except ValueError:
            auc_mean_c = float("nan")
        grid_summary[c] = float(auc_mean_c)
        if len(c_grid) > 1:
            print(f"    -> C={c}: AUC (promedio OOF entre repeticiones)={auc_mean_c:.3f}")
        if auc_mean_c > best_mean_auc:
            best_mean_auc = auc_mean_c
            best_c = c
            best_result = result

    if len(c_grid) > 1:
        print(f"[i] C elegido para el stacking final: {best_c} (AUC promedio={best_mean_auc:.3f}) "
              f"— elegido por CV, ver README sobre este sesgo de selección de hiperparámetro.")

    oof_score = best_result["oof_score_reps"].mean(axis=0)
    oof_prob = best_result["oof_prob_reps"].mean(axis=0)
    per_repeat_auc = best_result["per_repeat_auc"]
    valid_repeat_aucs = per_repeat_auc[~np.isnan(per_repeat_auc)]

    eta = choose_threshold_youden(oof_score, all_y)
    try:
        auc_cv = roc_auc_score(all_y, oof_score)
    except ValueError:
        auc_cv = float("nan")
    oof_pred = (oof_score > eta).astype(int)
    acc_cv = accuracy_score(all_y, oof_pred)
    cm_cv = confusion_matrix(all_y, oof_pred, labels=[0, 1])

    brier = brier_score(all_y, oof_prob)
    ece_report = expected_calibration_error(all_y, oof_prob, n_bins=args.ece_bins)
    auc_ci = bootstrap_auc_ci(all_y, oof_score, n_boot=args.n_bootstrap, seed=args.seed)

    print("\n===== REPORTE DE VALIDACIÓN CRUZADA REPETIDA (OOF, cubre el 100% de las llamadas en cada repetición) =====")
    print(f"  eta (umbral score LLR, elegido sobre el OOF promediado)  : {eta:.3f}")
    print(f"  AUC (sobre el score OOF promediado entre repeticiones)   : {auc_cv:.3f}")
    if auc_ci["ci_low"] is not None:
        print(f"  AUC bootstrap 95% CI (incertidumbre por tamaño de datos) : [{auc_ci['ci_low']:.3f}, {auc_ci['ci_high']:.3f}]")
    else:
        print("  AUC bootstrap 95% CI: no calculable (muy pocas llamadas de alguna clase)")
    print(f"  Accuracy (OOF, con eta de arriba)                        : {acc_cv:.3f}")
    if len(valid_repeat_aucs):
        print(f"  AUC entre REPETICIONES de CV (media ± std, {len(valid_repeat_aucs)} repeticiones): "
              f"{valid_repeat_aucs.mean():.3f} ± {valid_repeat_aucs.std():.3f}  <- estabilidad real del modelo")
        if valid_repeat_aucs.std() > 0.10:
            print("  [!] El AUC varía bastante entre repeticiones de CV: con este tamaño de dataset la métrica "
                  "todavía es inestable. Más llamadas (o más diversidad de generadores TTS) siguen ayudando más "
                  "que seguir ajustando el modelo.")
    print(f"  Brier score (calibración, menor=mejor)                   : {brier:.3f}")
    print(f"  ECE (calibración, menor=mejor)                           : {ece_report['ece']:.3f}")
    print(f"  Matriz de confusión (OOF)                                :\n{cm_cv}  (filas=real[human,synth], cols=pred)")

    # =========================================================
    # 2) MODELO FINAL: reajustado sobre el 100% de las llamadas
    #    (con augmentación incluida), pero con eta/C elegidos
    #    honestamente sobre la CV repetida.
    # =========================================================
    Xa_call_full = call_level_matrix(per_call, all_indices, "X_a", n_dim_a)
    Xb_call_full = call_level_matrix(per_call, all_indices, "X_b", n_dim_b)
    mask_a_final = select_feature_mask(Xa_call_full, all_y, args.feature_auc_margin, args.min_features_acoustic)
    mask_b_final = select_feature_mask(Xb_call_full, all_y, args.feature_auc_margin, args.min_features_behavioral)

    detector = VoiceAuthenticityDetector()
    detector.prior_h1 = args.prior_h1
    detector.acoustic_block = fit_block(per_call, all_indices, "X_a", ACOUSTIC_FEATURE_NAMES, n_dim_a,
                                         mask_a_final, include_augmented=use_augment)
    detector.behavioral_block = fit_block(per_call, all_indices, "X_b", BEHAVIORAL_FEATURE_NAMES, n_dim_b,
                                           mask_b_final, include_augmented=False)

    final_llr_a, final_llr_b, final_y = training_llr_pairs(
        per_call, all_indices, detector.acoustic_block, detector.behavioral_block, include_augmented=use_augment)
    detector.calibrator = StackingCalibrator(C=best_c)
    detector.calibrator.fit(final_llr_a, final_llr_b, final_y)
    detector.eta = eta

    detector.train_report = {
        "n_calls_total": len(calls),
        "n_folds": n_folds,
        "n_repeats": args.n_repeats,
        "augmentation": {
            "enabled": use_augment,
            "n_variants_per_call": args.n_augments,
            "total_extra_segments": int(total_aug_segments),
            "note": "aumento solo de condiciones de canal/entorno sobre llamadas reales; "
                    "nunca se usan variantes aumentadas para validar, solo para entrenar cada fold",
        },
        "eta": detector.eta,
        "stacking_C": best_c,
        "stacking_C_grid_results": grid_summary if len(c_grid) > 1 else None,
        "oof_auc": float(auc_cv),
        "oof_auc_bootstrap_ci95": [auc_ci["ci_low"], auc_ci["ci_high"]],
        "oof_accuracy": float(acc_cv),
        "oof_confusion_matrix": cm_cv.tolist(),
        "oof_brier_score": float(brier),
        "oof_ece": float(ece_report["ece"]),
        "cv_repeat_auc_mean": float(valid_repeat_aucs.mean()) if len(valid_repeat_aucs) else None,
        "cv_repeat_auc_std": float(valid_repeat_aucs.std()) if len(valid_repeat_aucs) else None,
        "cv_repeat_aucs": per_repeat_auc.tolist(),
        "last_repeat_fold_reports": best_result["last_fold_reports"],
        "feature_keep_frequency_acoustic": {
            n: float(f) for n, f in zip(ACOUSTIC_FEATURE_NAMES, best_result["feature_keep_freq_a"])
        },
        "feature_keep_frequency_behavioral": {
            n: float(f) for n, f in zip(BEHAVIORAL_FEATURE_NAMES, best_result["feature_keep_freq_b"])
        },
        "n_features_acoustic_kept_final": int(mask_a_final.sum()),
        "n_features_behavioral_kept_final": int(mask_b_final.sum()),
        "acoustic_features_kept": [n for n, m in zip(ACOUSTIC_FEATURE_NAMES, mask_a_final) if m],
        "behavioral_features_kept": [n for n, m in zip(BEHAVIORAL_FEATURE_NAMES, mask_b_final) if m],
        "n_silence_segments_discarded": int(total_discarded),
        "stacking_weights": detector.calibrator.weights,
    }

    print(f"\n[i] Modelo final ajustado sobre el 100% de las {len(calls)} llamadas"
          f"{' (+ variantes aumentadas)' if use_augment else ''}.")
    print(f"[i] Features acústicas activas en el LLR final: {mask_a_final.sum()}/{n_dim_a} -> "
          f"{detector.train_report['acoustic_features_kept']}")
    print(f"[i] Features comportamentales activas en el LLR final: {mask_b_final.sum()}/{n_dim_b} -> "
          f"{detector.train_report['behavioral_features_kept']}")
    print(f"[i] Pesos stacking (final, C={best_c}): {detector.calibrator.weights}")
    unstable_a = [n for n, f in detector.train_report["feature_keep_frequency_acoustic"].items() if 0.2 < f < 0.8]
    if unstable_a:
        print(f"[i] Features acústicas cuya selección es inestable entre folds/repeticiones "
              f"(se activan/desactivan según la partición): {unstable_a}")

    if args.plots_dir:
        train_mask_plot = np.zeros(len(calls), dtype=bool)  # todas las llamadas son "OOF" aquí
        final_llr_a_per_call, final_llr_b_per_call = per_call_llrs(
            per_call, all_indices, detector.acoustic_block, detector.behavioral_block)
        plot_paths = generate_all_plots(
            args.plots_dir, cm_cv, all_y, oof_score, auc_cv, detector.eta,
            final_llr_a_per_call, final_llr_b_per_call, all_y, train_mask_plot, detector.calibrator,
            repeat_aucs=valid_repeat_aucs, ece_report=ece_report,
        )
        print("\n[i] Gráficas de diagnóstico (basadas en scores OOF + CV repetida) guardadas en:")
        for p in plot_paths:
            print(f"    - {p}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(out_path))
    print(f"\n[✔] Modelo guardado en: {out_path}")


if __name__ == "__main__":
    main()
