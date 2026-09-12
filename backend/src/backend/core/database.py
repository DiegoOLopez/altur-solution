from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# La URL de conexión apunta a localhost porque el puerto 3306 del contenedor está expuesto
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://api_user:api_password@127.0.0.1:3306/audio_training_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()