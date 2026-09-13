"""
Script de seed para poblar la base de datos con audios de prueba.

Carga un conjunto de grabaciones simuladas en la tabla ``audios``
y en el bucket de MinIO (usando datos falsos de ejemplo). Estos
audios aparecen en el Review Hub (estado no_revisado) para que el
usuario pueda probar el flujo de revisión y entrenamiento sin
tener que realizar llamadas reales previamente.

Uso:
    uv run python -m backend.seed
"""
import uuid
from datetime import datetime, timedelta

from minio import Minio

from backend.core.database import SessionLocal, engine
from backend.models.audio import Audio, AudioStatus, Base


# Cliente MinIO para subir los archivos de prueba.
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False
)
BUCKET_NAME = "grabaciones"

# Crear el bucket si no existe (al iniciar desde cero en Docker).
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)


def run_seed():
    """
    Inserta audios de prueba simulando llamadas del último día.

    Los datos son ficticios y apuntan a archivos WAV inexistentes
    en MinIO (útiles solo para poblar la UI). En un entorno de
    desarrollo completo, este script también subiría archivos
    .wav de prueba al bucket.
    """
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Si ya hay audios, no hacemos nada para evitar duplicados.
    if db.query(Audio).first():
        print("La base de datos ya contiene audios. Saltando seed.")
        return

    now = datetime.utcnow()

    # Generar 5 llamadas simuladas con distintos scores y timestamps.
    test_audios = [
        {
            "title": "Llamada Sospechosa 1",
            "duration": 45.2,
            "is_synthetic": True,
            "confidence": 0.98,
            "score_total": 0.95,
            "offset_minutes": 10,
        },
        {
            "title": "Verificación Identidad",
            "duration": 120.5,
            "is_synthetic": False,
            "confidence": 0.99,
            "score_total": 0.12,
            "offset_minutes": 45,
        },
        {
            "title": "Voz Robotizada",
            "duration": 15.0,
            "is_synthetic": True,
            "confidence": 0.85,
            "score_total": 0.82,
            "offset_minutes": 120,
        },
        {
            "title": "Soporte Técnico",
            "duration": 300.0,
            "is_synthetic": False,
            "confidence": 0.70,
            "score_total": 0.45,
            "offset_minutes": 200,
        },
        {
            "title": "Fraude Deepfake",
            "duration": 60.1,
            "is_synthetic": True,
            "confidence": 0.99,
            "score_total": 0.99,
            "offset_minutes": 300,
        },
    ]

    for audio_data in test_audios:
        session_id = str(uuid.uuid4())
        # Clave ficticia en MinIO.
        storage_key = f"audios/seed_{session_id}.wav"

        audio = Audio(
            ref=session_id,
            title=audio_data["title"],
            storage_key=storage_key,
            duration=audio_data["duration"],
            sample_rate=16000,
            channels=1,
            is_synthetic=audio_data["is_synthetic"],
            split="live",
            confidence=audio_data["confidence"],
            score_total=audio_data["score_total"],
            status=AudioStatus.NO_REVISADO,
            created_at=now - timedelta(minutes=audio_data["offset_minutes"])
        )
        db.add(audio)

    db.commit()
    print("Base de datos inicializada con audios de prueba.")
    db.close()


if __name__ == "__main__":
    run_seed()
