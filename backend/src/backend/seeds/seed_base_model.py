"""
Script de seed para la tabla ``trained_models``.

Inserta un registro inicial en la base de datos que representa
el modelo preentrenado de fábrica que viene incluido en la carpeta
``ai_models/altur_detector_model.joblib``.

Este script se ejecuta al levantar la base de datos para que el
Review Hub (y los endpoints como POST /review/train/cancel) tengan
un modelo "base" sobre el cual trabajar desde el primer momento.
"""
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from backend.core.database import SessionLocal, engine, Base
from backend.models.trained_model import TrainedModel, ModelStatus

# Asegurarse de que las tablas estén creadas antes de insertar.
# En un entorno real se usaría Alembic; aquí creamos todo con SQLAlchemy.
Base.metadata.create_all(bind=engine)


def seed_base_model():
    """
    Inserta el modelo base en la tabla ``trained_models``.

    El modelo base es "Altur Wav2Vec2 Base" y se asume que
    su .joblib ya está en el disco o en MinIO.
    """
    db: Session = SessionLocal()
    try:
        # Verificar si la tabla ya tiene modelos para no duplicar
        # el seed inicial.
        existing = db.query(TrainedModel).first()
        if existing:
            print("El modelo base ya está registrado.")
            return

        print("Insertando el modelo base preentrenado...")
        base_model = TrainedModel(
            ref=str(uuid.uuid4()),
            name="Altur Wav2Vec2 Base",
            # Esta es la ruta donde se guardaría en MinIO, aunque
            # en el código actual del detector el modelo base se
            # carga directamente desde el disco (ai_models/).
            storage_key="modelos/altur_detector_model.joblib",
            n_samples_human=1000,
            n_samples_synthetic=1000,
            val_accuracy=0.98,
            val_auc=0.99,
            eta=0.5,
            train_method="base_pretrained",
            # Como es el modelo inicial, su estado ya es DISPONIBLE.
            status=ModelStatus.DISPONIBLE,
            train_report='{"step": "Modelo base de fábrica"}',
            created_at=datetime.utcnow()
        )
        db.add(base_model)
        db.commit()
        print(f"✅ Modelo base registrado con ID {base_model.id}")
    except Exception as e:
        print(f"❌ Error insertando modelo base: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_base_model()
