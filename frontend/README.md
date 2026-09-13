# AuraVoice - Frontend

El frontend de AuraVoice es una aplicación desarrollada en **React (Vite)**. Proporciona una interfaz rica y moderna para que los analistas y agentes interactúen con el sistema de detección de voz sintética.

## 🌟 Módulos Principales

El frontend consta de tres vistas o flujos clave:

1. **Análisis Forense (Batch):** Interfaz para cargar un archivo `.wav`. Muestra de forma progresiva la validación del formato (8kHz, 16-bit PCM, estéreo), y luego envía el payload codificado en Base64 al backend. Finalmente, despliega un *dashboard* detallado (Bento Grid) con el veredicto, forma de onda, espectrograma interactivo, latencias y la explicación textual (LLM).
2. **Llamada en Vivo (Streaming):** Una demostración interactiva que captura el audio directamente del micrófono del agente utilizando `AudioWorkletProcessor`. El audio se convierte a PCM mono y se transmite en bloques hacia un WebSocket (`/ws/detect`). Mientras la llamada ocurre, el componente renderiza el score de detección acumulado.
3. **Review Hub:** Consola administrativa para auditar las llamadas almacenadas (recuperadas de MinIO/MySQL). Permite al equipo clasificar grabaciones y lanzar el **reentrenamiento del modelo** para mejorar su efectividad.

## 📂 Estructura de Directorios

- `src/components/`: Componentes modulares (Navbar, Bento UI, Visores de Ondas/Espectrograma, Panel de Review).
- `src/utils/`: Funciones para conversión y manipulación de audio en el cliente (codificación de WAV, resampling, parsing PCM y codificación Base64) y clientes HTTP (endpoints).
- `src/App.jsx`: Gestor de estado principal y ruteo de la aplicación.
- `src/index.css` & `src/App.css`: Sistema de diseño moderno, variables CSS, variables de cristal (glassmorphism), y utilidades.

## ⚡ Comandos Útiles

```bash
# Instalar dependencias
npm install

# Iniciar el servidor de desarrollo
npm run dev

# Compilar para producción
npm run build

# Previsualizar compilación
npm run preview
```

## 🔌 Integración con la API

La aplicación se comunica con el backend mediante las funciones expuestas en `utils/audioUtils.js` y mediante una conexión WebSocket nativa en el componente `StreamingCallSimulator.jsx`.

- **POST `/detect_wav`**: Endpoint para análisis batch forense (usado cuando el usuario sube un archivo local).
- **WS `/ws/detect`**: Endpoint WebSocket de streaming constante en vivo.
- **GET, POST `/review/*`**: Subrutas para obtener llamadas guardadas, actualizar la clasificación y arrancar un reentrenamiento del modelo AI en base a los metadatos de las tablas.
