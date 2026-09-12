import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    Integer,
    String,
    Text,
)

from backend.core.database import Base


class ModelStatus(str, enum.Enum):
    ELIMINADO = "eliminado"
    DISPONIBLE = "disponible"
    TRAINING = "training"


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    ref = Column(
        String(255),
        nullable=False,
        unique=True,
    )

    name = Column(
        String(255),
        nullable=False,
    )

    storage_key = Column(
        String(500),
        nullable=False,
    )

    # Métricas del entrenamiento
    n_samples_human = Column(
        Integer,
        nullable=False,
        default=0,
    )

    n_samples_synthetic = Column(
        Integer,
        nullable=False,
        default=0,
    )

    val_accuracy = Column(
        Float,
        nullable=True,
    )

    val_auc = Column(
        Float,
        nullable=True,
    )

    eta = Column(
        Float,
        nullable=True,
    )

    # Método usado: 'online_update' o 'full_retrain'
    train_method = Column(
        String(50),
        nullable=False,
        default="online_update",
    )

    # Reporte completo del entrenamiento (JSON serializado)
    train_report = Column(
        Text,
        nullable=True,
    )

    status = Column(
        SQLEnum(ModelStatus),
        nullable=False,
        default=ModelStatus.TRAINING,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
