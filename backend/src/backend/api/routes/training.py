"""
Endpoint de reentrenamiento del modelo.

Flujo:
  1. Frontend manda POST /review/train con el nombre del nuevo modelo
  2. Se consultan los audios clasificados (status='revisado') de MySQL
  3. Se descargan los WAVs desde MinIO
  4. Se ejecuta el pipeline completo de fit_densities (reentrenamiento de 0)
  5. Se guarda el nuevo .pkl en MinIO
  6. Se registra el modelo en la tabla trained_models
  7. Se responde con las métricas del entrenamiento
"""
import io
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from minio import Minio
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from backend.core.database import SessionLocal, get_db
from backend.models.audio import Audio, AudioStatus
from backend.models.trained_model import TrainedModel, ModelStatus

# ==========================================================
# Asegurarnos de que el detector esté en el path
# ==========================================================
DETECTOR_ROOT = Path(__file__).resolve().parents[5] / "detector"
if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))

from features.acoustic import segment_features
from features.behavioral import behavioral_event_features
from preprocessing.diarization import diarize_stereo
from model.pipeline import (
    VoiceAuthenticityDetector,
    load_wav_any,
    split_channels,
    SEGMENT_SECONDS,
)
from model.calibration import choose_threshold_youden

# ==========================================================
# MinIO
# ==========================================================
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False,
)
BUCKET_NAME = "grabaciones"

# ==========================================================
# Router
# ==========================================================
router = APIRouter(
    prefix="/review",
    tags=["Training"],
)

training_executor = ThreadPoolExecutor(max_workers=1)


class TrainRequest(BaseModel):
    model_name: str


class TrainResponse(BaseModel):
    message: str
    model_id: int
    model_ref: str
    model_name: str
    storage_key: str
    n_samples_human: int
    n_samples_synthetic: int
    val_accuracy: float | None
    val_auc: float | None
    eta: float | None
    status: str


class TrainStartResponse(BaseModel):
    message: str
    model_name: str
    status: str


class TrainStatusResponse(BaseModel):
    model_id: int | None
    model_name: str | None
    status: str
    progress: int
    step: str
    n_samples_human: int
    n_samples_synthetic: int
    val_accuracy: float | None = None
    val_auc: float | None = None
    error: str | None = None


def serialize_training_status(model: TrainedModel | None) -> TrainStatusResponse:
    if model is None:
        return TrainStatusResponse(
            model_id=None,
            model_name=None,
            status="idle",
            progress=0,
            step="Sin entrenamiento activo",
            n_samples_human=0,
            n_samples_synthetic=0,
        )

    report = json.loads(model.train_report or "{}")
    default_step = (
        "Entrenamiento completado"
        if model.status == ModelStatus.DISPONIBLE
        else "Procesando entrenamiento"
    )
    return TrainStatusResponse(
        model_id=model.id,
        model_name=model.name,
        status=model.status.value,
        progress=int(report.get("progress", 100 if model.status == ModelStatus.DISPONIBLE else 0)),
        step=report.get("step", default_step),
        n_samples_human=model.n_samples_human,
        n_samples_synthetic=model.n_samples_synthetic,
        val_accuracy=model.val_accuracy,
        val_auc=model.val_auc,
        error=report.get("error"),
    )


def update_training_progress(db: Session, model: TrainedModel, progress: int, step: str):
    model.train_report = json.dumps({"progress": progress, "step": step})
    db.commit()


def training_was_cancelled(db: Session, model_id: int) -> bool:
    status = db.query(TrainedModel.status).filter(TrainedModel.id == model_id).scalar()
    return status == ModelStatus.ELIMINADO


@router.get("/train/status", response_model=TrainStatusResponse)
def training_status(db: Session = Depends(get_db)):
    model = (
        db.query(TrainedModel)
        .filter(TrainedModel.status.in_([
            ModelStatus.TRAINING,
            ModelStatus.DISPONIBLE,
            ModelStatus.ELIMINADO,
        ]))
        .order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc())
        .first()
    )
    return serialize_training_status(model)


@router.post("/train/cancel", response_model=TrainStatusResponse)
def cancel_training(db: Session = Depends(get_db)):
    model = (
        db.query(TrainedModel)
        .filter(TrainedModel.status == ModelStatus.TRAINING)
        .order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc())
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="No hay un entrenamiento activo.")

    report = json.loads(model.train_report or "{}")
    report["step"] = "Entrenamiento cancelado"
    report["error"] = "Cancelado por el usuario."
    model.train_report = json.dumps(report)
    model.status = ModelStatus.ELIMINADO
    db.commit()
    db.refresh(model)
    return serialize_training_status(model)


def run_training_in_thread(payload: TrainRequest):
    db = SessionLocal()
    try:
        train_model_job(payload, db)
    finally:
        db.close()


def download_wav_from_minio(storage_key: str) -> bytes:
    """Descarga un archivo WAV de MinIO y regresa los bytes."""
    response = minio_client.get_object(BUCKET_NAME, storage_key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    return data


def extract_call_features(wav_bytes: bytes):
    """Extrae features acústicas y comportamentales de un WAV."""
    signal, sr = load_wav_any(wav_bytes)
    caller, agent = split_channels(signal)

    # Diarización automática si es estéreo
    if agent is not None:
        turns = diarize_stereo(signal, sr)
    else:
        turns = []

    # Features acústicas por segmento de 1s
    seg_len = int(SEGMENT_SECONDS * sr)
    n_segments = max(len(caller) // seg_len, 0)
    X_a = (
        np.array(
            [
                segment_features(caller[i * seg_len : (i + 1) * seg_len], sr)
                for i in range(n_segments)
            ]
        )
        if n_segments
        else np.empty((0, 1))
    )

    # Features comportamentales
    behavioral = behavioral_event_features(turns) if turns else []
    X_b = (
        np.array([x for _, x in behavioral])
        if behavioral
        else np.empty((0, 1))
    )

    return X_a, X_b


@router.post("/train", response_model=TrainStartResponse, status_code=202)
def start_training(
    payload: TrainRequest,
    db: Session = Depends(get_db),
):
    model_name = payload.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="El nombre del modelo es obligatorio.")

    active_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.status == ModelStatus.TRAINING)
        .first()
    )
    if active_model:
        raise HTTPException(
            status_code=409,
            detail=f"Ya hay un entrenamiento activo: {active_model.name}.",
        )

    training_executor.submit(run_training_in_thread, TrainRequest(model_name=model_name))
    return TrainStartResponse(
        message=f"El entrenamiento de '{model_name}' comenzó en segundo plano.",
        model_name=model_name,
        status=ModelStatus.TRAINING.value,
    )


def train_model_job(
    payload: TrainRequest,
    db: Session = Depends(get_db),
):
    """Reentrena el modelo desde cero con los audios clasificados."""

    model_name = payload.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="El nombre del modelo es obligatorio.")

    # ==========================================================
    # 1. Consultar audios clasificados
    # ==========================================================
    reviewed_audios = (
        db.query(Audio)
        .filter(Audio.status == AudioStatus.REVISADO)
        .all()
    )

    if len(reviewed_audios) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos 4 muestras clasificadas para reentrenar. "
                   f"Actualmente hay {len(reviewed_audios)}.",
        )

    # Verificar que hay al menos 1 de cada clase
    n_human = sum(1 for a in reviewed_audios if not a.is_synthetic)
    n_synthetic = sum(1 for a in reviewed_audios if a.is_synthetic)

    if n_human == 0 or n_synthetic == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesita al menos 1 muestra de cada clase. "
                   f"Humanas: {n_human}, Sintéticas: {n_synthetic}.",
        )

    # ==========================================================
    # 2. Crear registro del modelo con status TRAINING
    # ==========================================================
    session_id = uuid.uuid4().hex
    storage_key = f"modelos/model_{session_id}.pkl"

    nuevo_modelo = TrainedModel(
        ref=session_id,
        name=model_name,
        storage_key=storage_key,
        n_samples_human=n_human,
        n_samples_synthetic=n_synthetic,
        train_method="full_retrain",
        status=ModelStatus.TRAINING,
        train_report=json.dumps({"progress": 0, "step": "Preparando datos"}),
        created_at=datetime.utcnow(),
    )
    db.add(nuevo_modelo)
    db.commit()
    db.refresh(nuevo_modelo)

    print(f"🏋️ Iniciando reentrenamiento: '{model_name}' (ref={session_id})")
    print(f"   Muestras: {n_human} humanas + {n_synthetic} sintéticas = {len(reviewed_audios)} total")

    try:
        # ==========================================================
        # 3. Descargar WAVs y extraer features
        # ==========================================================
        per_call = {}
        labels = []

        update_training_progress(db, nuevo_modelo, 5, "Descargando audios clasificados")

        for i, audio_record in enumerate(reviewed_audios):
            if training_was_cancelled(db, nuevo_modelo.id):
                return

            label = 1 if audio_record.is_synthetic else 0
            labels.append(label)

            print(f"   [{i + 1}/{len(reviewed_audios)}] Descargando {audio_record.storage_key}...")
            wav_bytes = download_wav_from_minio(audio_record.storage_key)

            X_a, X_b = extract_call_features(wav_bytes)
            per_call[i] = {
                "X_a": X_a,
                "X_b": X_b,
                "label": label,
                "title": audio_record.title,
            }
            update_training_progress(
                db,
                nuevo_modelo,
                5 + int(((i + 1) / len(reviewed_audios)) * 55),
                f"Procesando audio {i + 1} de {len(reviewed_audios)}",
            )

            print(
                f"       → {X_a.shape[0]} segmentos acústicos, "
                f"{X_b.shape[0]} eventos comportamentales"
            )

        # ==========================================================
        # 4. Split train/val
        # ==========================================================
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        labels_arr = np.array(labels)
        n_calls = len(reviewed_audios)

        # Si hay muy pocas muestras, usar todo para train y no hacer val
        if n_calls < 6:
            idx_train = list(range(n_calls))
            idx_val = list(range(n_calls))  # val = train (sin reportar métricas reales)
            print(f"   ⚠️ Pocas muestras ({n_calls}), usando todo para train")
        else:
            idx_train, idx_val = train_test_split(
                range(n_calls),
                test_size=0.25,
                random_state=42,
                stratify=labels,
            )

        print(f"   Split: train={len(idx_train)} val={len(idx_val)}")

        # ==========================================================
        # 5. Crear nuevo detector y ajustar densidades
        # ==========================================================
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        update_training_progress(db, nuevo_modelo, 65, "Ajustando detector")
        detector = VoiceAuthenticityDetector()
        detector.prior_h1 = 0.3

        # Agrupar features por clase (solo TRAIN)
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

        n_dim_a = next(
            (d["X_a"].shape[1] for d in per_call.values() if d["X_a"].shape[1] > 1),
            len(detector.acoustic_block.feature_names),
        )
        n_dim_b = next(
            (d["X_b"].shape[1] for d in per_call.values() if d["X_b"].shape[1] > 1),
            len(detector.behavioral_block.feature_names),
        )

        detector.acoustic_block.fit(
            stack_or_zeros(train_h0_a, n_dim_a),
            stack_or_zeros(train_h1_a, n_dim_a),
        )
        detector.behavioral_block.fit(
            stack_or_zeros(train_h0_b, n_dim_b),
            stack_or_zeros(train_h1_b, n_dim_b),
        )

        print(
            f"   Densidades: acústico h0={len(train_h0_a)} h1={len(train_h1_a)} | "
            f"comportamental h0={len(train_h0_b)} h1={len(train_h1_b)}"
        )

        # ==========================================================
        # 6. Calcular LLR por llamada y ajustar stacking
        # ==========================================================
        def call_llrs(d):
            llr_a = (
                sum(detector.acoustic_block.llr(x) for x in d["X_a"])
                if d["X_a"].shape[1] > 1
                else 0.0
            )
            llr_b = (
                sum(detector.behavioral_block.llr(x) for x in d["X_b"])
                if d["X_b"].shape[1] > 1
                else 0.0
            )
            return llr_a, llr_b

        all_llr_a, all_llr_b, all_y = [], [], []
        for i in range(n_calls):
            d = per_call[i]
            llr_a, llr_b = call_llrs(d)
            all_llr_a.append(llr_a)
            all_llr_b.append(llr_b)
            all_y.append(d["label"])

        all_llr_a = np.array(all_llr_a)
        all_llr_b = np.array(all_llr_b)
        all_y = np.array(all_y)

        train_mask = np.zeros(n_calls, dtype=bool)
        train_mask[list(idx_train)] = True

        # Ajustar stacking logístico sobre TRAIN
        detector.calibrator.fit(
            all_llr_a[train_mask],
            all_llr_b[train_mask],
            all_y[train_mask],
        )

        # ==========================================================
        # 7. Elegir umbral eta sobre VAL
        # ==========================================================
        val_scores = all_llr_a[~train_mask] + all_llr_b[~train_mask]
        val_y = all_y[~train_mask]

        if len(val_scores) > 0 and len(np.unique(val_y)) > 1:
            detector.eta = choose_threshold_youden(val_scores, val_y)
        else:
            # Fallback: usar 0
            detector.eta = 0.0

        # ==========================================================
        # 8. Métricas de validación
        # ==========================================================
        val_accuracy = None
        val_auc = None

        if len(val_scores) > 0 and len(np.unique(val_y)) > 1:
            val_pred = (val_scores > detector.eta).astype(int)
            try:
                val_auc = float(roc_auc_score(val_y, val_scores))
            except ValueError:
                val_auc = None
            val_accuracy = float(accuracy_score(val_y, val_pred))
            cm = confusion_matrix(val_y, val_pred, labels=[0, 1])
        else:
            cm = None

        train_report = {
            "n_calls_total": n_calls,
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_human": n_human,
            "n_synthetic": n_synthetic,
            "eta": detector.eta,
            "val_auc": val_auc,
            "val_accuracy": val_accuracy,
            "val_confusion_matrix": cm.tolist() if cm is not None else None,
            "stacking_weights": detector.calibrator.weights,
        }

        detector.train_report = train_report

        print(f"   eta={detector.eta:.3f}")
        if val_accuracy is not None:
            print(f"   accuracy={val_accuracy:.3f}")
        if val_auc is not None:
            print(f"   AUC={val_auc:.3f}")

        # ==========================================================
        # 9. Serializar y subir a MinIO
        # ==========================================================
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        update_training_progress(db, nuevo_modelo, 90, "Guardando modelo entrenado")
        pkl_buffer = io.BytesIO()
        joblib.dump(detector, pkl_buffer, compress=3)
        pkl_buffer.seek(0)
        file_size = pkl_buffer.getbuffer().nbytes

        minio_client.put_object(
            BUCKET_NAME,
            storage_key,
            pkl_buffer,
            length=file_size,
            content_type="application/octet-stream",
        )

        if training_was_cancelled(db, nuevo_modelo.id):
            return

        print(f"   ☁️ Modelo subido a MinIO: {storage_key} ({file_size} bytes)")

        # ==========================================================
        # 10. Actualizar registro en MySQL
        # ==========================================================
        nuevo_modelo.val_accuracy = val_accuracy
        nuevo_modelo.val_auc = val_auc
        nuevo_modelo.eta = detector.eta
        train_report["progress"] = 100
        train_report["step"] = "Entrenamiento completado"
        nuevo_modelo.train_report = json.dumps(train_report)
        nuevo_modelo.status = ModelStatus.DISPONIBLE
        db.commit()
        db.refresh(nuevo_modelo)

        print(f"   ✅ Modelo '{model_name}' entrenado y registrado (id={nuevo_modelo.id})")

        return TrainResponse(
            message=f"Modelo '{model_name}' entrenado exitosamente con {n_calls} muestras.",
            model_id=nuevo_modelo.id,
            model_ref=nuevo_modelo.ref,
            model_name=nuevo_modelo.name,
            storage_key=nuevo_modelo.storage_key,
            n_samples_human=n_human,
            n_samples_synthetic=n_synthetic,
            val_accuracy=val_accuracy,
            val_auc=val_auc,
            eta=detector.eta,
            status=nuevo_modelo.status.value,
        )

    except Exception as e:
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        # Si falla, marcamos el modelo como eliminado
        nuevo_modelo.status = ModelStatus.ELIMINADO
        nuevo_modelo.train_report = json.dumps({"progress": 0, "step": "Error", "error": str(e)})
        db.commit()

        print(f"   ❌ Error en reentrenamiento: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el reentrenamiento: {str(e)}",
        )
