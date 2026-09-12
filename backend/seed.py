# ese archivo se creó con la finalidad de llenar la base a datos con los audios y existentes en la carpeta "audios"
import csv
# Ajusta estas importaciones según la estructura de tu proyecto
from src.backend.core.database import SessionLocal 
# IMPORTANTE: Asegúrate de importar también AudioStatus
from src.backend.models import Audio, AudioStatus

def seed_audios():
    db = SessionLocal()
    
    try:
        with open("manifest.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            nuevos_audios = []
            for row in reader:
                is_synthetic = (row["label"] == "synthetic")
                
                nuevo_audio = Audio(
                    ref=row["anon_id"],
                    title=f"Call_{row['anon_id']}", 
                    storage_key=f"audios/{row['anon_id']}.wav",
                    duration=float(row["duration_s"]),
                    sample_rate=16000,
                    channels=1,
                    is_synthetic=is_synthetic,
                    split=row["split"],
                    # Los audios recién cargados deben esperar revisión:
                    status=AudioStatus.NO_REVISADO
                )
                nuevos_audios.append(nuevo_audio)
            
            db.add_all(nuevos_audios)
            db.commit()
            
            print(f"✅ ¡Éxito! Se insertaron {len(nuevos_audios)} audios con estado 'NO_REVISADO'.")
            
    except Exception as e:
        db.rollback()
        print(f"❌ Error al insertar datos: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_audios()