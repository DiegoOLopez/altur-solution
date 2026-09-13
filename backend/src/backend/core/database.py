"""
Configuración de la base de datos (SQLAlchemy).

Establece la conexión a MySQL vía PyMySQL y provee helpers para
inyectar una sesión de base de datos por request (FastAPI ``Depends``)
y cerrarla correctamente al terminar.

Componentes principales:
    - ``engine``       : motor de conexión SQLAlchemy.
    - ``SessionLocal`` : fábrica de sesiones para interactuar con la BD.
    - ``Base``         : clase declarativa base para los modelos ORM.
    - ``get_db()``     : dependencia de FastAPI que entrega y cierra sesiones.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# URL de conexión a MySQL.
# Apunta a localhost:3306 porque el contenedor de MySQL expone su puerto
# hacia el host en el entorno de desarrollo (docker-compose.yml).
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://api_user:api_password@127.0.0.1:3306/audio_training_db"

# Motor de conexión. Reutilizado por toda la aplicación.
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Fábrica de sesiones. autocommit=False obliga a hacer commit explícito
# para que los cambios se persistan; autoflush=False evita flushes
# automáticos que podrían causar errores inesperados.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base declarativa para definir modelos ORM.
# Todos los modelos (Audio, TrainedModel, etc.) heredan de Base.
Base = declarative_base()


def get_db():
    """
    Dependencia de FastAPI que entrega una sesión de base de datos
    y la cierra automáticamente al finalizar la request.

    Uso en un endpoint:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...

    Yields:
        Session: sesión de SQLAlchemy lista para queries.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()