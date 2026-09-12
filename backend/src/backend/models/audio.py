import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    Integer,
    String,
)

from backend.core.database import Base


class AudioStatus(str, enum.Enum):
    REVISADO = "revisado"
    NO_REVISADO = "no_revisado"
    DELETED = "deleted"


class ModelStatus(str, enum.Enum):
    ELIMINADO = "eliminado"
    DISPONIBLE = "disponible"
    TRAINING = "training"

class Audio(Base):
    __tablename__ = "audios"

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

    title = Column(
        String(255),
        nullable=False,
    )

    storage_key = Column(
        String(500),
        nullable=False,
    )

    duration = Column(
        Float,
        nullable=False,
    )

    sample_rate = Column(
        Integer,
        nullable=False,
    )

    channels = Column(
        Integer,
        nullable=False,
    )

    is_synthetic = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    split = Column(
        String(20),
        nullable=False,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    score_total = Column(
        Float,
        nullable=True,
    )

    status = Column(
        SQLEnum(AudioStatus),
        nullable=False,
        default=AudioStatus.NO_REVISADO,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )