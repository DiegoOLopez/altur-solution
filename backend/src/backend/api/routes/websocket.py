from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(
    tags=["WebSocket"],
)


@router.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()

            # Aquí posteriormente irá la lógica de detección
            await websocket.send_json(
                {
                    "message": "Detection WebSocket",
                    "received": data,
                }
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected")