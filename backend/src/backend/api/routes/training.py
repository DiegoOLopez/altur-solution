"""
Endpoint de reentrenamiento del modelo.

Flujo completo:
  1. Frontend manda POST /review/train con el nombre del nuevo modelo.
  2. Se consultan los audios clasificados (status='revisado') de MySQL.
  3. Se descargan los WAVs desde MinIO.
  4. Se extraen features con Wav2Vec2 (capa oculta 4, promedio temporal).
  5. Se entrena un clasificador LogisticRegression.
  6. Se guarda el nuevo .joblib en MinIO.
  7. Se registra el modelo en la tabla trained_models.
  8. Se responde con las métricas del entrenamiento.

El entrenamiento se ejecuta en un hilo separado (ThreadPoolExecutor)
para no bloquear el event loop de FastAPI. El frontend puede consultar
el progreso en tiempo real con GET /review/train/status.

Endpoints:
    POST /review/train         — Iniciar reentrenamiento (202 Accepted).
    GET  /review/train/status   — Consultar progreso del entrenamiento.
    POST /review/train/cancel   — Cancelar un entrenamiento en curso.
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
from fastapi.responses import StreamingResponse
from minio import Minio
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from backend.core.database import SessionLocal, get_db
from backend.models.audio import Audio, AudioStatus
from backend.models.trained_model import TrainedModel, ModelStatus

# ==========================================================
# Configuración del path al módulo detector
# ==========================================================
# El código del detector original se encuentra en la carpeta
# ``detector/`` en la raíz del repositorio. Se agrega al sys.path
# para poder importar sus módulos sin instalarlo como paquete.
DETECTOR_ROOT = Path(__file__).resolve().parents[5] / "detector"
if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))

import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==========================================================
# Cliente MinIO para acceso al almacenamiento de objetos
# ==========================================================
# Las credenciales corresponden al entorno de desarrollo
# definido en docker-compose.yml.
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False,
)

# Nombre del bucket donde se almacenan grabaciones y modelos.
BUCKET_NAME = "grabaciones"

# ==========================================================
# Router y executor para entrenamiento en background
# ==========================================================
router = APIRouter(
    prefix="/review",
    tags=["Training"],
)

# ThreadPoolExecutor con un solo worker para asegurar que solo
# se ejecute un entrenamiento a la vez.
training_executor = ThreadPoolExecutor(max_workers=1)


# ==========================================================
# Schemas de request/response para el endpoint de entrenamiento
# ==========================================================

class TrainRequest(BaseModel):
    """Cuerpo de la petición para iniciar un reentrenamiento."""
    model_name: str


class TrainResponse(BaseModel):
    """Respuesta completa tras finalizar un reentrenamiento exitoso."""
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
    """Respuesta inmediata (HTTP 202) al iniciar el entrenamiento."""
    message: str
    model_name: str
    status: str


class TrainStatusResponse(BaseModel):
    """Estado actual del entrenamiento (usado por el frontend para polling)."""
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


# ==========================================================
# Funciones auxiliares
# ==========================================================

def serialize_training_status(model: TrainedModel | None) -> TrainStatusResponse:
    """
    Convierte un registro de TrainedModel en la representación
    que consume el frontend para mostrar el progreso del entrenamiento.

    Args:
        model: Instancia del modelo ORM, o None si no hay entrenamiento.

    Returns:
        TrainStatusResponse con el estado actual.
    """
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

    # Parsear el reporte JSON almacenado en la BD.
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
    """
    Actualiza el progreso del entrenamiento en la BD para que
    el frontend pueda consultar el estado en tiempo real.

    Args:
        db:       Sesión de base de datos.
        model:    Instancia del modelo ORM.
        progress: Porcentaje de progreso (0-100).
        step:     Descripción textual del paso actual.
    """
    model.train_report = json.dumps({"progress": progress, "step": step})
    db.commit()


def training_was_cancelled(db: Session, model_id: int) -> bool:
    """
    Verifica si el entrenamiento fue cancelado por el usuario.

    Se consulta el status del modelo en la BD; si fue marcado como
    ELIMINADO, el hilo de entrenamiento debe detenerse.

    Args:
        db:       Sesión de base de datos.
        model_id: ID del modelo en entrenamiento.

    Returns:
        True si el entrenamiento fue cancelado.
    """
    status = db.query(TrainedModel.status).filter(TrainedModel.id == model_id).scalar()
    return status == ModelStatus.ELIMINADO


# ==========================================================
# Endpoints
# ==========================================================

@router.get("/train/status", response_model=TrainStatusResponse)
def training_status(db: Session = Depends(get_db)):
    """
    Consulta el estado del entrenamiento más reciente.

    El frontend hace polling a este endpoint para actualizar la barra
    de progreso y el paso actual del reentrenamiento.

    Returns:
        TrainStatusResponse con el estado del último modelo.
    """
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
    """
    Cancela un entrenamiento en curso marcando el modelo como ELIMINADO.

    El hilo de entrenamiento verifica periódicamente si fue cancelado
    y se detiene cuando detecta el cambio de status.

    Returns:
        TrainStatusResponse con el estado actualizado.

    Raises:
        HTTPException 404 si no hay un entrenamiento activo.
    """
    model = (
        db.query(TrainedModel)
        .filter(TrainedModel.status == ModelStatus.TRAINING)
        .order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc())
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="No hay un entrenamiento activo.")

    # Marcar como cancelado y guardar el motivo en el reporte.
    report = json.loads(model.train_report or "{}")
    report["step"] = "Entrenamiento cancelado"
    report["error"] = "Cancelado por el usuario."
    model.train_report = json.dumps(report)
    model.status = ModelStatus.ELIMINADO
    db.commit()
    db.refresh(model)
    return serialize_training_status(model)


@router.get("/models")
def list_trained_models(db: Session = Depends(get_db)):
    """
    Lista todos los modelos entrenados disponibles.
    """
    models = (
        db.query(TrainedModel)
        .filter(TrainedModel.status == ModelStatus.DISPONIBLE)
        .order_by(TrainedModel.created_at.desc())
        .all()
    )
    
    return [
        {
            "id": m.id,
            "name": m.name,
            "created_at": m.created_at,
            "n_samples_human": m.n_samples_human,
            "n_samples_synthetic": m.n_samples_synthetic,
            "val_accuracy": m.val_accuracy,
            "val_auc": m.val_auc,
        }
        for m in models
    ]


@router.get("/models/{model_id}/download")
def download_trained_model(model_id: int, db: Session = Depends(get_db)):
    """
    Descarga el archivo .joblib de un modelo entrenado desde MinIO.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model or model.status != ModelStatus.DISPONIBLE:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no disponible.")
    
    try:
        response = minio_client.get_object(BUCKET_NAME, model.storage_key)
        
        # Generator for streaming the file chunk by chunk
        def iterfile():
            try:
                for chunk in response.stream(32 * 1024):
                    yield chunk
            finally:
                response.close()
                response.release_conn()

        filename = model.storage_key.split("/")[-1]
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        return StreamingResponse(iterfile(), media_type="application/octet-stream", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al descargar el modelo: {str(e)}")


def run_training_in_thread(payload: TrainRequest):
    """
    Wrapper para ejecutar el entrenamiento en un hilo separado.

    Crea su propia sesión de BD (no puede reutilizar la de FastAPI
    porque corre en un thread distinto) y la cierra al terminar.

    Args:
        payload: TrainRequest con el nombre del modelo a entrenar.
    """
    db = SessionLocal()
    try:
        train_model_job(payload, db)
    finally:
        db.close()


# ==========================================================
# Funciones de procesamiento de audio y features
# ==========================================================

def download_wav_from_minio(storage_key: str) -> bytes:
    """
    Descarga un archivo WAV de MinIO y regresa los bytes crudos.

    Args:
        storage_key: Clave del objeto en el bucket de MinIO.

    Returns:
        Bytes del archivo WAV completo.
    """
    response = minio_client.get_object(BUCKET_NAME, storage_key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    return data


def extract_wav2vec_features(wav_bytes: bytes, processor, model, device):
    """
    Extrae features de Wav2Vec2 a partir de los bytes de un archivo WAV.

    Proceso:
    1. Lee el WAV con soundfile.
    2. Si es estéreo, toma solo el canal 0 (llamante).
    3. Resamplea a 16 kHz si es necesario.
    4. Trunca a 3 segundos (48,000 muestras) para uniformidad.
    5. Pasa por Wav2Vec2 y extrae la capa oculta 4.
    6. Promedia temporalmente para obtener un vector de 768 dimensiones.

    Args:
        wav_bytes: Bytes del archivo WAV.
        processor: Instancia de Wav2Vec2Processor.
        model:     Instancia de Wav2Vec2Model en el device correcto.
        device:    torch.device (cpu, cuda, o mps).

    Returns:
        numpy array de shape (768,) con las features extraídas.
    """
    # Leer audio con soundfile (soporta múltiples formatos PCM).
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')

    # Si es estéreo, tomar solo el canal del llamante (canal 0).
    if data.ndim > 1:
        data = data[:, 0]

    # Convertir a tensor de PyTorch con shape [1, num_samples].
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Resamplear a 16 kHz si el audio tiene otra frecuencia.
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Truncar a 3 segundos (48,000 muestras a 16 kHz) para uniformidad.
    if waveform.shape[1] > 48000:
        waveform = waveform[:, :48000]

    # Procesar con el tokenizer de Wav2Vec2.
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extraer features de la capa oculta 4 (sin gradientes).
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[4]
        # Promedio temporal: [1, T, 768] -> [768]
        features = hidden_states.mean(dim=1).squeeze().cpu().numpy()

    return features


@router.post("/train", response_model=TrainStartResponse, status_code=202)
def start_training(
    payload: TrainRequest,
    db: Session = Depends(get_db),
):
    """
    Inicia el reentrenamiento del modelo en segundo plano.

    Valida que no haya otro entrenamiento activo y lanza el job
    en un ThreadPoolExecutor. Retorna HTTP 202 (Accepted) inmediatamente.

    Args:
        payload: TrainRequest con el nombre del modelo.
        db:      Sesión de base de datos.

    Returns:
        TrainStartResponse con el estado inicial.

    Raises:
        HTTPException 400 si el nombre está vacío.
        HTTPException 409 si ya hay un entrenamiento activo.
    """
    model_name = payload.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="El nombre del modelo es obligatorio.")

    # Verificar que no haya un entrenamiento en curso.
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

    # Lanzar el entrenamiento en un hilo separado.
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
    """
    Job principal de reentrenamiento. Se ejecuta en un hilo separado.

    Pasos:
    1. Consultar audios clasificados en la BD.
    2. Crear registro del modelo con status TRAINING.
    3. Descargar WAVs y extraer features Wav2Vec2.
    4. Dividir en train/val.
    5. Entrenar LogisticRegression.
    6. Calcular métricas de validación.
    7. Serializar y subir el modelo a MinIO.
    8. Actualizar registro en MySQL con métricas y status DISPONIBLE.

    Args:
        payload: TrainRequest con el nombre del modelo.
        db:      Sesión de base de datos (creada por el thread).
    """

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

    # Se necesitan al menos 4 muestras para poder hacer train/val split.
    if len(reviewed_audios) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos 4 muestras clasificadas para reentrenar. "
                   f"Actualmente hay {len(reviewed_audios)}.",
        )

    # Verificar que hay al menos 1 muestra de cada clase.
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
    storage_key = f"modelos/model_{session_id}.joblib"

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
        # 3. Descargar WAVs y extraer features Wav2Vec2
        # ==========================================================
        # Seleccionar el mejor dispositivo disponible (MPS para Mac, CUDA, o CPU).
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        wav2vec_model.eval()

        # Diccionario para almacenar features por índice de audio.
        per_call = {}
        labels = []

        update_training_progress(db, nuevo_modelo, 5, "Descargando audios clasificados")

        for i, audio_record in enumerate(reviewed_audios):
            # Verificar cancelación antes de procesar cada audio.
            if training_was_cancelled(db, nuevo_modelo.id):
                return

            # Etiquetar: 1 = sintético, 0 = humano.
            label = 1 if audio_record.is_synthetic else 0
            labels.append(label)

            print(f"   [{i + 1}/{len(reviewed_audios)}] Descargando {audio_record.storage_key}...")
            wav_bytes = download_wav_from_minio(audio_record.storage_key)

            # Extraer features Wav2Vec2 (vector de 768 dimensiones).
            features = extract_wav2vec_features(wav_bytes, processor, wav2vec_model, device)
            per_call[i] = {
                "features": features,
                "label": label,
                "title": audio_record.title,
            }

            # Actualizar progreso: de 5% a 60% conforme se procesan los audios.
            update_training_progress(
                db,
                nuevo_modelo,
                5 + int(((i + 1) / len(reviewed_audios)) * 55),
                f"Procesando audio {i + 1} de {len(reviewed_audios)}",
            )

        # ==========================================================
        # 4. Split train/val
        # ==========================================================
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        labels_arr = np.array(labels)
        n_calls = len(reviewed_audios)

        # Si hay muy pocas muestras, usar todo para train (sin val real).
        if n_calls < 6:
            idx_train = list(range(n_calls))
            idx_val = list(range(n_calls))  # val = train (métricas orientativas)
            print(f"   ⚠️ Pocas muestras ({n_calls}), usando todo para train")
        else:
            # Split estratificado 75/25 para mantener proporción de clases.
            idx_train, idx_val = train_test_split(
                range(n_calls),
                test_size=0.25,
                random_state=42,
                stratify=labels,
            )

        print(f"   Split: train={len(idx_train)} val={len(idx_val)}")

        # ==========================================================
        # 5. Entrenar clasificador LogisticRegression
        # ==========================================================
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        update_training_progress(db, nuevo_modelo, 65, "Ajustando detector")

        # Construir matrices de features y labels para entrenamiento.
        X_train = np.array([per_call[i]["features"] for i in idx_train])
        y_train = np.array([per_call[i]["label"] for i in idx_train])

        # LogisticRegression con class_weight='balanced' para manejar
        # desbalanceo entre clases humano/sintético.
        detector = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
        detector.fit(X_train, y_train)

        # ==========================================================
        # 6. Métricas de validación
        # ==========================================================
        val_accuracy = None
        val_auc = None

        if len(idx_val) > 0 and len(np.unique(labels_arr[idx_val])) > 1:
            X_val = np.array([per_call[i]["features"] for i in idx_val])
            y_val = np.array([per_call[i]["label"] for i in idx_val])

            # Probabilidades de la clase positiva (sintético).
            val_scores = detector.predict_proba(X_val)[:, 1]
            val_pred = detector.predict(X_val)

            try:
                val_auc = float(roc_auc_score(y_val, val_scores))
            except ValueError:
                val_auc = None
            val_accuracy = float(accuracy_score(y_val, val_pred))
            cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
        else:
            cm = None

        # Umbral de decisión por defecto (0.5 para LogisticRegression).
        detector.eta = 0.5

        # Reporte completo del entrenamiento (se guarda como JSON en la BD).
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
        }

        print(f"   eta={detector.eta:.3f}")
        if val_accuracy is not None:
            print(f"   accuracy={val_accuracy:.3f}")
        if val_auc is not None:
            print(f"   AUC={val_auc:.3f}")

        # ==========================================================
        # 7. Serializar y subir el modelo a MinIO
        # ==========================================================
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        update_training_progress(db, nuevo_modelo, 90, "Guardando modelo entrenado")

        # Serializar el clasificador con joblib (compresión nivel 3).
        pkl_buffer = io.BytesIO()
        joblib.dump(detector, pkl_buffer, compress=3)
        pkl_buffer.seek(0)
        file_size = pkl_buffer.getbuffer().nbytes

        # Subir el modelo serializado al bucket de MinIO.
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
        # 8. Actualizar registro en MySQL
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
        # Verificar si fue cancelado durante la excepción.
        if training_was_cancelled(db, nuevo_modelo.id):
            return

        # Si falla, marcar el modelo como eliminado con el error.
        nuevo_modelo.status = ModelStatus.ELIMINADO
        nuevo_modelo.train_report = json.dumps({"progress": 0, "step": "Error", "error": str(e)})
        db.commit()

        print(f"   ❌ Error en reentrenamiento: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el reentrenamiento: {str(e)}",
        )
