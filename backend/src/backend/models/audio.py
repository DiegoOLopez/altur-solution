from sqlalchemy import Column, Integer, String, Boolean, Enum as SQLEnum
import enum
from src.backend.core.database import Base

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
    id = Column(Integer, primary_key=True, index=True)
    ref = Column(String(255), nullable=False)
    is_syntetic = Column(Boolean, default=False)
    status = Column(SQLEnum(AudioStatus), default=AudioStatus.NO_REVISADO)

class Modelo(Base):
    __tablename__ = "modelos"
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    estado = Column(SQLEnum(ModelStatus), default=ModelStatus.TRAINING)
    referencia = Column(String(255), nullable=False)