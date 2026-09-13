# Altur Solution: AuraVoice

> Sistema avanzado de detección de voz sintética (Deepfakes) para protección de contact centers bancarios, desarrollado para el reto de Tecnologías Altur en HackMTY26.

![AuraVoice Dashboard](https://img.shields.io/badge/AuraVoice-Detect-0284c7?style=flat-square&logo=react)
![FastAPI Backend](https://img.shields.io/badge/Backend-FastAPI-059669?style=flat-square&logo=fastapi)
![Wav2Vec2 Model](https://img.shields.io/badge/AI_Model-Wav2Vec2-ff3366?style=flat-square&logo=pytorch)

AuraVoice es una solución integral que protege las operaciones telefónicas bancarias detectando voces sintéticas, clonadas o deepfakes en tiempo real y diferido. Utiliza el modelo preentrenado Wav2Vec2 de Meta para extraer características acústicas complejas, junto con un clasificador que aprende iterativamente gracias a un ciclo de revisión humana (Human-in-the-Loop).

## 🚀 Características Principales

1. **Detección Forense (Batch):** Sube audios `.wav` para un análisis exhaustivo donde se evalúan características espectrales, latencia de interrupción y micro-respiraciones de la llamada (Endpoint HTTP POST).
2. **Streaming en Vivo:** Simulación de detección en tiempo real. Envía bloques PCM mediante WebSockets y obtiene un veredicto provisional cada 3 segundos, lo que permite interrumpir llamadas fraudulentas de inmediato.
3. **Review Hub y Reentrenamiento (Active Learning):** Los agentes pueden escuchar grabaciones almacenadas, clasificarlas como "sintéticas" o "reales", y lanzar un job en background para **reentrenar el modelo** y adaptarse a nuevas tácticas de evasión.
4. **Cumplimiento de la Norma Altur:** El sistema procesa y normaliza el audio telefónico estéreo estándar (8,000 Hz, 16-bit PCM).

## 📂 Arquitectura del Proyecto

Este monorepo contiene tanto el frontend como el backend. Las aplicaciones se comunican principalmente mediante REST (para lote y revisión) y WebSockets (para streaming).

- [**`/frontend`**](./frontend/README.md): Interfaz de usuario (React/Vite). Dashboard táctico forense, simulador en vivo y consola de revisión.
- [**`/backend`**](./backend/README.md): Servidor API (FastAPI) y motor de inferencia/streaming. Administra la conexión a MySQL (metadatos de audios) y MinIO (almacenamiento de `.wav` y `.joblib`).
- **`/detector`**: Librería base que define la red neuronal y extrae características de Wav2Vec2. Se enlaza dinámicamente con el backend.

## 🛠 Instalación y Arranque Rápido

El proyecto está dockerizado para asegurar una infraestructura predecible (Base de datos MySQL y almacenamiento MinIO).

### Requisitos
- Node.js (v18+)
- Python (v3.10+) con `uv`
- Docker y Docker Compose

### 1. Iniciar la Infraestructura Base (MySQL y MinIO)
```bash
cd backend
docker-compose up -d
```
> Esto levantará la base de datos MySQL (`audio_training_db`) y el bucket MinIO de forma local.

### 2. Iniciar el Backend (API)
```bash
cd backend
uv run uvicorn src.backend.main:app --reload
```
> En otra terminal, corre el script de seed para inyectar algunos audios de prueba en el hub de revisión: `uv run python -m src.backend.seed`

### 3. Iniciar el Frontend
```bash
cd frontend
npm install
npm run dev
```

Abra el navegador en `http://localhost:5173` para usar AuraVoice.

---
**Altur Solution** - Creado para proteger la identidad y seguridad bancaria en la era de la inteligencia artificial.
