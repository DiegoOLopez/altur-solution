"""
Script de utilidad para probar rápidamente el endpoint POST /detect.

Envía un payload JSON con un audio codificado en Base64, simulando
la estructura exacta que usaba el juez externo durante el hackathon.
"""
import base64
import json

import requests


def main():
    """
    Envía una petición al endpoint de detección offline (batch).

    1. Carga un archivo WAV local (``sample.wav``).
    2. Lo codifica en Base64.
    3. Forma el JSON ``DetectionJsonRequest``.
    4. Envía el POST a http://localhost:8000/detect.
    """
    url = "http://localhost:8000/detect"

    # Intentar cargar un archivo de prueba en la raíz del proyecto.
    # En un caso real, el script espera que este archivo exista.
    try:
        with open("sample.wav", "rb") as f:
            audio_bytes = f.read()
    except FileNotFoundError:
        print("Error: No se encontró 'sample.wav'.")
        print("Coloque un archivo sample.wav a 8kHz estéreo en el directorio actual para probar.")
        return

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "call_id": "test_123",
        "audio_base64": b64_audio,
        "sample_rate": 8000,
        "channels": 2
    }

    print(f"Sending POST request to {url}...")
    response = requests.post(url, json=payload)

    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
