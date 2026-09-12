import json
import uuid
import wave
import io
from datetime import datetime

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends
)
from minio import Minio
from sqlalchemy.orm import Session

# Ajusta estas importaciones a las rutas de tu proyecto
from backend.services.audio_normalizer import AudioNormalizer
from backend.services.detector import AudioDetector
from backend.core.database import get_db
from backend.models.audio import Audio, AudioStatus  # Ajusta la ruta a donde guardaste el modelo

router = APIRouter(tags=["WebSocket"])

# ==========================================================
# Configuración del Cliente MinIO
# Como el backend corre local (uv) y MinIO en Docker, apuntamos a localhost:9000
# ==========================================================
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False  # True si usaras HTTPS
)
BUCKET_NAME = "grabaciones"

@router.websocket("/ws/detect")
async def detect_websocket(
    websocket: WebSocket, 
    db: Session = Depends(get_db) # Inyectamos la conexión a MySQL
):
    await websocket.accept()

    audio_buffer = bytearray()
    detection_history = []

    try:
        # ... [TODO TU CÓDIGO DE VALIDACIÓN DE JSON (PASOS 1, 2 Y 3) SE QUEDA EXACTAMENTE IGUAL] ...
        
        # Simulando que pasaste la validación y tienes sample_rate
        message = await websocket.receive_text()
        config = json.loads(message)
        sample_rate = config.get("sample_rate")
        
        normalizer = AudioNormalizer(sample_rate)
        detector = AudioDetector()
        
        await websocket.send_json({"event": "ready", "input_sample_rate": sample_rate, "output_sample_rate": 16000, "channels": 1, "sample_width": 2, "converted": sample_rate == 8000, "window_ms": 1000, "hop_ms": 100})

        chunk_number = 0

        while True:
            audio_chunk = await websocket.receive_bytes()
            audio_buffer.extend(audio_chunk)
            chunk_number += 1

            normalized_audio = normalizer.process(audio_chunk)
            snapshot = detector.process_audio(normalized_audio, 16_000)

            await websocket.send_json({"event": "chunk_processed", "chunk": chunk_number, "input_bytes": len(audio_chunk), "output_bytes": len(normalized_audio), "detection_ready": snapshot is not None})

            if snapshot is not None:
                detection_history.append(snapshot) # Guardamos el dict del snapshot
                await websocket.send_json({"event": "detection", "result": snapshot})

    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected - Guardando en MinIO y MySQL...")
        
        if len(audio_buffer) > 0:
            session_id = uuid.uuid4().hex
            
            # ==========================================================
            # A. Preparar el WAV en Memoria RAM (sin tocar disco)
            # ==========================================================
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_buffer)
            
            wav_io.seek(0)
            file_size = wav_io.getbuffer().nbytes
            
            # ==========================================================
            # B. Subir a MinIO
            # ==========================================================
            storage_key = f"audios/call_{session_id}.wav"
            
            minio_client.put_object(
                BUCKET_NAME,
                storage_key,
                wav_io,
                length=file_size,
                content_type="audio/wav"
            )
            print(f"☁️ Audio subido a MinIO: {storage_key}")

            # ==========================================================
            # C. Calcular métricas para la Base de Datos
            # ==========================================================
            # 1. Duración en segundos = total_bytes / (sample_rate * 2 bytes por sample)
            duration_sec = len(audio_buffer) / (sample_rate * 2)
            
            # 2. Promediar los scores de la llamada
            avg_confidence = 0.0
            avg_score = 0.0
            is_synth = False
            
            if detection_history:
                # Ajusta las claves según lo que devuelva tu `snapshot`
                avg_confidence = sum(d.get("confidence", 0) for d in detection_history) / len(detection_history)
                avg_score = sum(d.get("score_total", 0) for d in detection_history) / len(detection_history)
                # Si el score promedio pasa cierto umbral, es sintético
                is_synth = avg_score > 0.5 

            # ==========================================================
            # D. Insertar Registro en MySQL
            # ==========================================================
            nuevo_audio = Audio(
                ref=session_id,
                title=f"Demo Live {datetime.utcnow().strftime('%H:%M:%S')}",
                storage_key=storage_key,  # La ruta para encontrarlo en MinIO
                duration=duration_sec,
                sample_rate=sample_rate,
                channels=1,
                is_synthetic=is_synth,
                split="live", # Identificador para audios grabados en el hackathon
                confidence=avg_confidence,
                score_total=avg_score,
                status=AudioStatus.NO_REVISADO
            )
            
            db.add(nuevo_audio)
            db.commit()
            print("💾 Metadata guardada en MySQL.")