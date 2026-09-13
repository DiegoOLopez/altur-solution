"""
Modelo ORM ``Audio``: representa una grabación telefónica en la base de datos.

Cada registro guarda los metadatos de una grabación procesada por el detector
(duración, tasa de muestreo, canales, resultado, confianza) y su ubicación
en MinIO (``storage_key``). El campo ``status`` controla el ciclo de vida
dentro del Review Hub.

Tabla: ``audios``
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
)

from backend.core.database import Base


class AudioStatus(str, enum.Enum):
    """
    Estado del ciclo de vida de una grabación en el Review Hub.

    Valores:
        NO_REVISADO : el audio no ha sido clasificado por un humano.
        REVISADO    : un humano lo clasificó como real o sintético.
        DELETED     : fue marcado para eliminación lógica (no se borra de MinIO).
    """
    NO_REVISADO = "no_revisado"
    REVISADO = "revisado"
    DELETED = "deleted"


class Audio(Base):
    """
    Registro de una grabación telefónica procesada por el sistema.

    Columnas:
        id           : PK autoincremental.
        ref          : UUID corto que identifica la grabación externamente.
        title        : nombre amigable (p.ej. el nombre del archivo).
        storage_key  : clave del objeto en MinIO (bucket ``grabaciones``).
        duration     : duración en segundos.
        sample_rate  : frecuencia de muestreo del audio original.
        channels     : número de canales (1 mono, 2 estéreo).
        is_synthetic : ``True`` si el detector clasificó la voz como sintética.
        split        : partición de datos (p.ej. ``train``, ``val``, ``live``).
        confidence   : confianza del modelo (0.0 a 1.0).
        score_total  : score acumulado (LLR total del detector).
        status       : estado en el ciclo de vida del Review Hub.
        created_at   : timestamp de creación.
    """

    __tablename__ = "audios"

    id = Column(Integer, primary_key=True, index=True)
    ref = Column(String(36), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    storage_key = Column(String(512), nullable=False)
    duration = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    is_synthetic = Column(Integer, nullable=False)
    split = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True, default=0.0)
    score_total = Column(Float, nullable=True, default=0.0)
    status = Column(
        Enum(AudioStatus),
        nullable=False,
        default=AudioStatus.NO_REVISADO,
    )
    created_at = Column(DateTime, default=datetime.utcnow)