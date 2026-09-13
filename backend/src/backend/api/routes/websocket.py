"""
WebSocket /ws/detect para el análisis de voz sintética en tiempo real.

Protocolo:
1. El cliente envía un mensaje JSON de configuración con ``sample_rate``.
2. El servidor responde con un evento ``ready`` confirmando los parámetros.
3. El cliente envía chunks de PCM mono en binario (16-bit, sample_rate Hz).
4. Por cada chunk, el servidor:
   a. Normaliza a 16 kHz (si es necesario).
   b. Alimenta el detector streaming.
   c. Responde con ``chunk_processed`` y, si hay suficiente audio
      acumulado (~3s), con ``detection`` incluyendo el snapshot parcial.
5. Al desconectarse, la llamada completa se persiste:
   a. El WAV se construye en memoria y se sube a MinIO (bucket ``grabaciones``).
   b. Las métricas se guardan en MySQL para revisarlas desde el Review Hub.
"""
import io
import json
import uuid
import wave
from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    WebSocket,
    WebSocketDisconnect,
)
from minio import Minio
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.audio import Audio, AudioStatus
from backend.services.audio_normalizer import AudioNormalizer
from backend.services.detector import AudioDetector

router = APIRouter(tags=["WebSocket"])

# Cliente MinIO para persistir las grabaciones de streaming.
# El backend corre localmente y MinIO en Docker, por lo que
# apuntamos a localhost:9000. ``secure=False`` corresponde al desarrollo.
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False,
)

# Nombre del bucket donde se almacenan las grabaciones.
BUCKET_NAME = "grabaciones"


@router.websocket("/ws/detect")
async def detect_websocket(
    websocket: WebSocket,
    db: Session = Depends(get_db),
):
    """
    Endpoint WebSocket para detección de voz sintética en tiempo real.

    Flujo:
    1. Acepta la conexión WebSocket.
    2. Recibe configuración JSON (sample_rate).
    3. Recibe chunks de audio PCM binario en un loop.
    4. Normaliza cada chunk a 16 kHz y lo pasa al detector streaming.
    5. Cuando el detector acumula suficiente audio (~3s), envía un
       snapshot con el veredicto parcial.
    6. Al desconectarse, construye un WAV en memoria, lo sube a MinIO
       y guarda los metadatos en MySQL.

    Args:
        websocket: Conexión WebSocket activa.
        db:        Sesión de base de datos inyectada por FastAPI.
    """
    await websocket.accept()

    # Buffer para acumular todos los chunks y poder construir
    # el WAV completo al final de la llamada.
    audio_buffer = bytearray()

    # Historial de snapshots de detección para calcular métricas
    # promediadas al finalizar la llamada.
    detection_history = []

    try:
        # --- Paso 1: Recibir configuración del cliente ---
        # El cliente envía primero la configuración de la captura:
        # sample_rate (8000 o 16000 Hz), canales y ancho de muestra.
        message = await websocket.receive_text()
        config = json.loads(message)
        sample_rate = config.get("sample_rate")

        # Crear normalizador que convierte el audio a 16 kHz si es necesario.
        normalizer = AudioNormalizer(sample_rate)

        # Crear instancia del detector para esta sesión.
        detector = AudioDetector()

        # Confirmar al cliente que estamos listos.
        await websocket.send_json({
            "event": "ready",
            "input_sample_rate": sample_rate,
            "output_sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "converted": sample_rate == 8000,
            "window_ms": 1000,
            "hop_ms": 100,
        })

        chunk_number = 0

        # --- Paso 2: Loop de recepción de chunks ---
        while True:
            # Recibir chunk de audio PCM binario del cliente.
            audio_chunk = await websocket.receive_bytes()
            audio_buffer.extend(audio_chunk)
            chunk_number += 1

            # Normalizar a 16 kHz y pasar al detector.
            normalized_audio = normalizer.process(audio_chunk)
            snapshot = detector.process_audio(normalized_audio, 16_000)

            # Notificar al cliente que el chunk fue procesado.
            await websocket.send_json({
                "event": "chunk_processed",
                "chunk": chunk_number,
                "input_bytes": len(audio_chunk),
                "output_bytes": len(normalized_audio),
                "detection_ready": snapshot is not None,
            })

            # Si el detector tiene un snapshot (acumuló ~3s de audio),
            # enviar los resultados parciales al cliente.
            if snapshot is not None:
                detection_history.append(snapshot)
                await websocket.send_json(
                    {"event": "detection", "result": snapshot}
                )

    except WebSocketDisconnect:
        # --- Paso 3: Persistir la llamada al desconectarse ---
        print(
            "WebSocket disconnected - saving audio and metadata "
            "to MinIO and MySQL..."
        )

        if len(audio_buffer) > 0:
            session_id = uuid.uuid4().hex

            # A. Construir el WAV en memoria (sin tocar disco).
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_buffer)

            wav_io.seek(0)
            file_size = wav_io.getbuffer().nbytes

            # B. Subir el audio a MinIO.
            storage_key = f"audios/call_{session_id}.wav"

            minio_client.put_object(
                BUCKET_NAME,
                storage_key,
                wav_io,
                length=file_size,
                content_type="audio/wav",
            )
            print(f"Audio uploaded to MinIO: {storage_key}")

            # C. Calcular las métricas promediadas de la llamada.
            # Duración en segundos = total_bytes / (sample_rate * 2 bytes por muestra).
            duration_sec = len(audio_buffer) / (sample_rate * 2)

            # Promediar los scores obtenidos durante la llamada.
            avg_confidence = 0.0
            avg_score = 0.0
            is_synth = False

            if detection_history:
                avg_confidence = (
                    sum(d.get("confidence", 0) for d in detection_history)
                    / len(detection_history)
                )
                avg_score = (
                    sum(d.get("score_total", 0) for d in detection_history)
                    / len(detection_history)
                )
                # Si el score promedio pasa el umbral 0.5, clasificar como sintética.
                is_synth = avg_score > 0.5

            # D. Insertar el registro en MySQL para el Review Hub.
            nuevo_audio = Audio(
                ref=session_id,
                title=f"Demo Live {datetime.utcnow().strftime('%H:%M:%S')}",
                storage_key=storage_key,
                duration=duration_sec,
                sample_rate=sample_rate,
                channels=1,
                is_synthetic=is_synth,
                split="live",
                confidence=avg_confidence,
                score_total=avg_score,
                status=AudioStatus.NO_REVISADO,
            )

            db.add(nuevo_audio)
            db.commit()
            print("Metadata saved to MySQL.")