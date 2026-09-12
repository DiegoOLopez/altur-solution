"""
WebSocket /ws/detect para el análisis de voz sintética en tiempo real.

El cliente inicia la sesión enviando primero un mensaje JSON con la
configuración de captura (``sample_rate``). Después envía chunks de PCM
mono en binario; por cada chunk, el servidor lo normaliza a 16 kHz, lo
alimenta al detector streaming y responde con el snapshot acumulado de
la evidencia.

Al desconectarse, la llamada completa se persiste:

1. El WAV de la llamada se construye y se sube a MinIO (bucket
   ``grabaciones``) sin tocar disco.
2. Las métricas (confianza y score promedio, plus el veredicto) se
   guardan en MySQL para poder revisarlas después desde el Review Hub.
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

# Configuración del cliente MinIO.
# El backend corre con uv localmente y MinIO en Docker, por lo que
# apuntamos a localhost:9000. ``secure=False`` corresponde al desarrollo.
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="supersecretpassword",
    secure=False,
)
BUCKET_NAME = "grabaciones"


@router.websocket("/ws/detect")
async def detect_websocket(
    websocket: WebSocket,
    db: Session = Depends(get_db),
):
    await websocket.accept()

    audio_buffer = bytearray()
    detection_history = []

    try:
        # El cliente envía primero la configuración de la captura:
        # sample_rate (8000 o 16000 Hz), canales y ancho de muestra.
        message = await websocket.receive_text()
        config = json.loads(message)
        sample_rate = config.get("sample_rate")

        normalizer = AudioNormalizer(sample_rate)
        detector = AudioDetector()

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

        while True:
            audio_chunk = await websocket.receive_bytes()
            audio_buffer.extend(audio_chunk)
            chunk_number += 1

            normalized_audio = normalizer.process(audio_chunk)
            snapshot = detector.process_audio(normalized_audio, 16_000)

            await websocket.send_json({
                "event": "chunk_processed",
                "chunk": chunk_number,
                "input_bytes": len(audio_chunk),
                "output_bytes": len(normalized_audio),
                "detection_ready": snapshot is not None,
            })

            if snapshot is not None:
                # Guardamos el snapshot para calcular métricas al final.
                detection_history.append(snapshot)
                await websocket.send_json(
                    {"event": "detection", "result": snapshot}
                )

    except WebSocketDisconnect:
        print(
            "WebSocket disconnected - saving audio and metadata "
            "to MinIO and MySQL..."
        )

        if len(audio_buffer) > 0:
            session_id = uuid.uuid4().hex

            # A. Preparar el WAV en memoria (sin tocar disco).
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

            # C. Calcular las métricas de la llamada.
            # 1. Duración en segundos = total_bytes / (sample_rate * 2).
            duration_sec = len(audio_buffer) / (sample_rate * 2)

            # 2. Promediar los scores obtenidos durante la llamada.
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
                # Si el score promedio pasa el umbral, la llamada es sintética.
                is_synth = avg_score > 0.5

            # D. Insertar el registro en MySQL.
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