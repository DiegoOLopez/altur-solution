"""
Modelo ORM ``TrainedModel``: registra cada modelo entrenado por el sistema.

Cada vez que se ejecuta un reentrenamiento (POST /review/train), se crea un
registro con los metadatos del modelo resultante: nombre, ubicación en MinIO,
número de muestras, métricas de validación y estado.

Tabla: ``trained_models``
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
)

from backend.core.database import Base


class ModelStatus(str, enum.Enum):
    """
    Estado del ciclo de vida de un modelo entrenado.

    Valores:
        TRAINING   : el entrenamiento está en curso.
        DISPONIBLE : el modelo fue entrenado exitosamente y está listo.
        ELIMINADO  : el modelo fue cancelado o marcado para eliminación.
    """
    TRAINING = "training"
    DISPONIBLE = "disponible"
    ELIMINADO = "eliminado"


class TrainedModel(Base):
    """
    Registro de un modelo entrenado por el sistema de reentrenamiento.

    Columnas:
        id                 : PK autoincremental.
        ref                : UUID corto que identifica el modelo.
        name               : nombre dado por el usuario al iniciar el entreno.
        storage_key        : clave del .joblib en MinIO (bucket ``grabaciones``).
        n_samples_human    : cantidad de muestras humanas usadas.
        n_samples_synthetic: cantidad de muestras sintéticas usadas.
        val_accuracy       : accuracy en el conjunto de validación.
        val_auc            : AUC-ROC en el conjunto de validación.
        eta                : umbral de decisión del clasificador.
        train_method       : método de entrenamiento (p.ej. ``full_retrain``).
        train_report       : JSON con el reporte completo del entrenamiento.
        status             : estado actual del modelo.
        created_at         : timestamp de creación.
    """

    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    ref = Column(String(36), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    storage_key = Column(String(512), nullable=False)
    n_samples_human = Column(Integer, default=0)
    n_samples_synthetic = Column(Integer, default=0)
    val_accuracy = Column(Float, nullable=True)
    val_auc = Column(Float, nullable=True)
    eta = Column(Float, nullable=True)
    train_method = Column(String(50), nullable=True)
    train_report = Column(Text, nullable=True)
    status = Column(
        Enum(ModelStatus),
        nullable=False,
        default=ModelStatus.TRAINING,
    )
    created_at = Column(DateTime, default=datetime.utcnow)
