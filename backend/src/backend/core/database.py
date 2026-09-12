"""
Configuración de la base de datos (SQLAlchemy).

Conexión a MySQL vía PyMySQL y helpers para inyectar una sesión por
request (FastAPI ``Depends``) y cerrarla correctamente al terminar.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# La URL apunta a localhost porque el puerto 3306 del contenedor de
# MySQL está expuesto hacia el host en el entorno de desarrollo.
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://api_user:api_password@127.0.0.1:3306/audio_training_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    Dependencia que entrega una sesión de base de datos y la cierra
    automáticamente al finalizar la request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()