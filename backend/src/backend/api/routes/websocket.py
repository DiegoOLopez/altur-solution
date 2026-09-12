import json

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from backend.services.audio_normalizer import (
    AudioNormalizer,
)
from backend.services.sliding_window import (
    SlidingWindow,
)


router = APIRouter(
    tags=["WebSocket"],
)


@router.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        # ==========================================================
        # 1. Esperamos configuración inicial
        # ==========================================================

        message = await websocket.receive_text()

        try:
            config = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Invalid JSON configuration.",
                }
            )
            await websocket.close(code=1003)
            return

        sample_rate = config.get("sample_rate")
        channels = config.get("channels")
        sample_width = config.get("sample_width")

        # ==========================================================
        # 2. Validar formato
        # ==========================================================

        if sample_rate not in (8000, 16000):
            await websocket.send_json(
                {
                    "event": "error",
                    "message": (
                        "sample_rate must be 8000 or 16000."
                    ),
                }
            )
            await websocket.close(code=1003)
            return

        if channels != 1:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Only mono audio is supported.",
                }
            )
            await websocket.close(code=1003)
            return

        if sample_width != 2:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": (
                        "Only 16-bit PCM audio is supported."
                    ),
                }
            )
            await websocket.close(code=1003)
            return

        # ==========================================================
        # 3. Crear pipeline
        # ==========================================================

        normalizer = AudioNormalizer(sample_rate)
        sliding_window = SlidingWindow()

        await websocket.send_json(
            {
                "event": "ready",
                "input_sample_rate": sample_rate,
                "output_sample_rate": 16_000,
                "channels": 1,
                "sample_width": 2,
                "converted": sample_rate == 8000,
                "window_ms": 1000,
                "hop_ms": 100,
            }
        )

        # ==========================================================
        # 4. Recibir audio
        # ==========================================================

        while True:
            audio_chunk = await websocket.receive_bytes()

            # ------------------------------------------------------
            # Normalizar
            # ------------------------------------------------------

            normalized_audio = normalizer.process(
                audio_chunk
            )

            # ------------------------------------------------------
            # Ventana deslizante
            # ------------------------------------------------------

            windows = sliding_window.add(
                normalized_audio
            )

            # ------------------------------------------------------
            # Informar recepción
            # ------------------------------------------------------

            await websocket.send_json(
                {
                    "event": "chunk_processed",
                    "input_bytes": len(audio_chunk),
                    "output_bytes": len(normalized_audio),
                    "windows_ready": len(windows),
                }
            )

            # ------------------------------------------------------
            # Procesar ventanas
            # ------------------------------------------------------

            for window in windows:
                await websocket.send_json(
                    {
                        "event": "window_ready",
                        "window": sliding_window.window_number,
                        "bytes": len(window),
                        "duration_ms": 1000,
                    }
                )

                # ==================================================
                # AQUÍ ENTRA EL MODELO
                # ==================================================
                #
                # result = detector.predict(window)
                #
                # await websocket.send_json(
                #     {
                #         "event": "detection",
                #         "result": result,
                #     }
                # )

    except WebSocketDisconnect:
        print("WebSocket disconnected")