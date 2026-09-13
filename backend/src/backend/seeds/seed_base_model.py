"""
Registra el modelo base (altur_detector_model.joblib) en la tabla trained_models.

Uso:
    uv run python -m src.backend.seeds.seed_base_model
"""
import uuid
from datetime import datetime

from backend.core.database import SessionLocal
from backend.models.trained_model import TrainedModel, ModelStatus


def seed():
    db = SessionLocal()

    # Verificar si ya existe un modelo base
    existing = db.query(TrainedModel).filter(
        TrainedModel.name == "Vocalis / base-01"
    ).first()

    if existing:
        print(f"⚠️  El modelo base ya existe (id={existing.id}, ref={existing.ref})")
        db.close()
        return

    base_model = TrainedModel(
        ref=uuid.uuid4().hex,
        name="Vocalis / base-01",
        storage_key="modelos/altur_detector_model.joblib",
        n_samples_human=0,
        n_samples_synthetic=0,
        val_accuracy=None,
        val_auc=None,
        eta=None,
        train_method="initial",
        train_report=None,
        status=ModelStatus.DISPONIBLE,
        created_at=datetime.utcnow(),
    )

    db.add(base_model)
    db.commit()
    db.refresh(base_model)

    print(f"✅ Modelo base registrado: id={base_model.id}, ref={base_model.ref}")
    print(f"   name: {base_model.name}")
    print(f"   storage_key: {base_model.storage_key}")
    print(f"   status: {base_model.status}")

    db.close()


if __name__ == "__main__":
    seed()
