# Frontend

## Descripción

Frontend React para interactuar con Altur/AuraVoice, una interfaz de detección de voz sintética en llamadas telefónicas. Permite preparar y analizar audios, visualizar evidencia acústica y comportamental, operar una llamada en streaming y revisar audios clasificados.

La aplicación mantiene el estado de la sesión en React. La persistencia de resultados, audios revisados y modelos entrenados corresponde al backend.

## Características

- Selector de tres modos: `Audio Forensics`, `Revisar notas` y `Live Call`.
- Carga de archivos `.wav`, `.mp3`, `.ogg` y `.flac` en el flujo forense.
- Validación de WAV nativo y conversión de audio a WAV estéreo, 8 kHz y 16-bit PCM en el navegador.
- Separación visual de Canal 0, llamante/cliente, y Canal 1, agente.
- Conversión de buffers WAV a Base64 para inspección y preparación del payload.
- Envío del archivo WAV al backend mediante `multipart/form-data`.
- Visualización del pipeline de ingesta, validación e inferencia.
- Visualización del veredicto humano vs. sintético, confianza y evidencia del modelo.
- Osciloscopio estéreo con reproducción local del audio cargado.
- Espectrograma del Canal 0 calculado en el navegador mediante STFT.
- Paneles de señales del modelo y evidencia conversacional.
- Historial de detecciones de la sesión actual.
- Captura de micrófono en Live Call mediante `getUserMedia`, `AudioContext` y `AudioWorklet`.
- Estado de conexión WebSocket, frecuencia de muestreo, chunks enviados y detección recibida.
- Revisión de audios pendientes con reproducción desde el backend.
- Selección previa de `Sintético`, `Humano` o `Eliminar`, con persistencia únicamente al confirmar.
- Estadísticas de audios pendientes, revisados y disponibles para entrenamiento.
- Inicio de entrenamiento en segundo plano, consulta periódica de progreso y cancelación.

## Tecnologías

- React `^19.2.8` y React DOM `^19.2.8`.
- Vite `^8.3.0`.
- `lucide-react` para iconos.
- Tailwind CSS `^4.3.3` y `@tailwindcss/postcss`.
- PostCSS y Autoprefixer.
- Oxlint.
- JavaScript y JSX. No hay TypeScript configurado.
- Web APIs: `fetch`, `WebSocket`, `MediaRecorder`, `MediaDevices`, `AudioContext`, `OfflineAudioContext`, `AudioWorklet` y Canvas 2D.

## Requisitos

- Node.js compatible con las versiones instaladas por npm.
- npm.
- Navegador con soporte para Web Audio API, `getUserMedia` y `AudioWorklet` para grabación y Live Call.
- Backend de Altur disponible, por defecto en `http://localhost:8000`.
- Para Live Call, el backend debe exponer el WebSocket `/ws/detect`.
- Para revisión y entrenamiento, deben estar disponibles los endpoints `/review/*` del backend.

El repositorio no fija una versión de Node.js en `package.json`.

## Instalación

Desde `frontend/`:

```bash
npm install
```

También puede usarse el lockfile existente:

```bash
npm ci
```

## Variables de entorno

La única variable de entorno utilizada por el frontend es:

| Variable | Propósito | Obligatoria | Ejemplo |
| --- | --- | --- | --- |
| `VITE_API_BASE_URL` | URL base del backend HTTP y del backend WebSocket | No | `http://localhost:8000` |

Si no se define, la aplicación usa `http://localhost:8000`.

El proyecto incluye [.env.example](.env.example):

```env
VITE_API_BASE_URL=http://localhost:8000
```

Vite expone al cliente las variables que comienzan con `VITE_`.

## Ejecución

Servidor de desarrollo:

```bash
npm run dev
```

El script ejecuta Vite en el puerto `5173` y acepta conexiones del host mediante `--host`.

Build de producción:

```bash
npm run build
```

Previsualización del build:

```bash
npm run preview
```

Lint:

```bash
npm run lint
```

## Arquitectura

La entrada está en `src/main.jsx`. Importa `index.css` y `App.css`, monta `App` dentro de `StrictMode` y usa el elemento `#root` de `index.html`.

No existe un router de páginas ni un estado global externo. `App.jsx` mantiene el modo activo, el audio seleccionado, el resultado de análisis, el estado de carga y el historial de la sesión. `Navbar.jsx` cambia entre `batch`, `review` y `streaming`.

### Componentes principales

- `App.jsx`: composición, estado de sesión, llamadas al detector y montaje condicional de modos.
- `Navbar.jsx`: marca, estado general y selector de modos.
- `BatchInput.jsx`: carga, normalización, pipeline y botón de detección.
- `ForensicPipeline.jsx`: etapas de ingesta e inferencia.
- `StereoWaveform.jsx`: Canvas de los dos canales y reproducción local.
- `Spectrogram.jsx`: Canvas del espectrograma del Canal 0.
- `DetectionSignals.jsx`: evidencia acústica y métricas del modelo.
- `ResponseTiming.jsx`: evidencia conversacional y eventos comportamentales.
- `VerdictPanel.jsx`: resultado, confianza, métricas y explicación.
- `CallHistory.jsx`: historial temporal de resultados de la sesión.
- `StreamingCallSimulator.jsx`: captura de micrófono y comunicación WebSocket.
- `ReviewHub.jsx`: cola de revisión, clasificación, estadísticas y entrenamiento.
- `AudioHub.jsx`: componente de entrada con presets, micrófono y archivo; existe en el código, pero el flujo actual de `App.jsx` usa `BatchInput.jsx` para el modo forense.

### Utilidades

`src/utils/audioUtils.js` contiene la codificación WAV estéreo 8 kHz/16-bit, resampling, conversión ArrayBuffer/Base64, inspección de metadatos WAV, lectura PCM, cálculo de espectrograma mediante FFT/STFT y formateo de bytes.

También contiene un generador de escenarios demo y un helper HTTP para streaming que existen como utilidades, aunque el flujo actual de `App.jsx` no los usa para sustituir respuestas reales.

## Flujo de detección

### Audio Forensics

```text
Usuario
  -> selecciona un archivo
  -> BatchInput inspecciona el formato
  -> si hace falta, decodifica y resamplea a 8 kHz estéreo 16-bit PCM
  -> genera un WAV y prepara Base64
  -> pulsa "EJECUTAR DETECCIÓN FORENSE"
  -> callDetectApi convierte Base64 a bytes
  -> POST /detect como multipart/form-data, campo file
  -> backend responde con veredicto y evidencia
  -> App actualiza paneles, gráficas e historial de sesión
```

El Canal 0 se trata como llamante/cliente y el Canal 1 como agente. En una fuente mono, el frontend duplica la fuente para formar el segundo canal.

### Live Call

```text
Usuario
  -> pulsa "Empezar a grabar"
  -> el navegador solicita permiso de micrófono
  -> se abre WebSocket /ws/detect
  -> se envía sample_rate, channels y sample_width
  -> AudioWorklet produce chunks PCM Int16
  -> los chunks se envían por WebSocket
  -> el frontend recibe eventos JSON, incluido detection
  -> StreamingCallSimulator actualiza conexión y detección
```

El componente usa `/ws/detect`; no realiza actualmente una petición HTTP a `/detect_streaming` desde este flujo.

### Revisión y entrenamiento

```text
ReviewHub
  -> GET /review/audios y GET /review/stats
  -> muestra audios pendientes y estadísticas
  -> GET /review/audios/{id}/stream para reproducir
  -> PATCH /review/audios/{id}/classification al confirmar
  -> DELETE /review/audios/{id} al confirmar eliminar
  -> POST /review/train inicia entrenamiento en segundo plano
  -> GET /review/train/status consulta progreso cada 2 segundos
  -> POST /review/train/cancel cancela el entrenamiento activo
```

## Integración con el Backend

La URL base se obtiene de `VITE_API_BASE_URL` o, en su ausencia, de `http://localhost:8000`.

### Detección batch

```http
POST /detect
Content-Type: multipart/form-data
```

El campo enviado se llama `file` y contiene `audio.wav`. El frontend espera una respuesta JSON con campos como:

```json
{
  "is_synthetic": false,
  "confidence": 0.98,
  "score_total": 0.12,
  "llr_acoustic_cum": 0.1,
  "llr_behavioral_cum": 0.02,
  "n_acoustic_segments": 4,
  "n_behavioral_events": 2,
  "eta": 0.0
}
```

El frontend mide la latencia, conserva la respuesta en `apiResponse` y presenta los valores en la UI. Si la petición falla, muestra un estado de error y no sustituye el resultado por un veredicto simulado en `App.jsx`.

### Streaming

La URL HTTP se transforma de `http` a `ws` y se agrega `/ws/detect`. Primero se envía:

```json
{
  "sample_rate": 16000,
  "channels": 1,
  "sample_width": 2
}
```

Después se envían buffers binarios PCM. El frontend procesa eventos JSON como `ready`, `chunk_processed`, `detection` y `error`.

### Revisión

```http
GET /review/audios
GET /review/stats
GET /review/audios/{id}/stream
PATCH /review/audios/{id}/classification
DELETE /review/audios/{id}
GET /review/train/status
POST /review/train
POST /review/train/cancel
```

La clasificación se envía así:

```json
{
  "classification": "synthetic"
}
```

Las opciones no modifican la base hasta que el usuario pulsa `Confirmar`. Las respuestas no exitosas se muestran como mensajes de error en `ReviewHub`.

## Estructura del proyecto

```text
frontend/
├── .env.example
├── index.html
├── package.json
├── package-lock.json
├── postcss.config.js
├── tailwind.config.js
├── vite.config.js
├── public/
│   ├── audio-processor.js
│   ├── favicon.svg
│   └── icons.svg
└── src/
    ├── App.jsx
    ├── App.css
    ├── index.css
    ├── main.jsx
    ├── components/
    │   ├── BatchInput.jsx
    │   ├── CallHistory.jsx
    │   ├── DetectionSignals.jsx
    │   ├── ForensicPipeline.jsx
    │   ├── Navbar.jsx
    │   ├── ResponseTiming.jsx
    │   ├── ReviewHub.jsx
    │   ├── Spectrogram.jsx
    │   ├── StereoWaveform.jsx
    │   ├── StreamingCallSimulator.jsx
    │   └── VerdictPanel.jsx
    └── utils/
        └── audioUtils.js
```

## UI/UX

- La navegación superior permite cambiar de modo sin cambiar de página.
- Audio Forensics muestra estados de ingesta y análisis mediante pipelines visuales.
- El botón de detección se deshabilita durante el procesamiento.
- La interfaz muestra archivo activo, tamaño aproximado de Base64, canales y estados del detector.
- El veredicto muestra clasificación, confianza y evidencia acústica/conversacional cuando existe una respuesta.
- Live Call muestra conexión WebSocket, grabación, sample rate, chunks y detección.
- La revisión muestra carga, cola vacía, errores, confirmación de clasificación y progreso de entrenamiento.
- La reproducción forense se realiza localmente con Web Audio; las notas revisables usan el endpoint de streaming del backend.

## Testing

No hay una suite de tests automatizados configurada en `package.json`. La validación disponible es:

```bash
npm run lint
```

## Build y producción

Generar el bundle:

```bash
npm run build
```

El resultado se genera en `dist/` mediante Vite. Para previsualizarlo:

```bash
npm run preview
```

No hay configuración adicional de despliegue o servidor estático dentro del frontend.

## Troubleshooting

### Backend no disponible

Verifica `VITE_API_BASE_URL`, el puerto del backend y que `POST /detect` o las rutas `/review/*` estén disponibles.

### CORS

El backend debe permitir el origen desde el que se sirve Vite. La configuración CORS pertenece al backend.

### Micrófono

Live Call y la grabación de `AudioHub` requieren permiso para `getUserMedia`. Revisa los permisos del navegador y las condiciones de seguridad que el navegador exige para acceder al micrófono.

### Formato de audio

Audio Forensics admite `.wav`, `.mp3`, `.ogg` y `.flac` en el selector. El frontend convierte el audio a WAV PCM estéreo de 8 kHz y 16-bit antes de enviarlo.

### WebSocket

Live Call construye `ws://` o `wss://` a partir de `VITE_API_BASE_URL` y conecta a `/ws/detect`. Comprueba que esa ruta esté disponible y acepte audio mono PCM de 16-bit.

### Tailwind/PostCSS

El proyecto usa Tailwind 4. `postcss.config.js` debe conservar `@tailwindcss/postcss` e `index.css` debe importar Tailwind con `@import "tailwindcss";`.

### Lint

`npm run lint` puede mostrar advertencias existentes en otros componentes. El proyecto no configura esas advertencias como una suite de tests.

## Limitaciones

- No hay router de páginas ni estado global persistente en el frontend.
- El historial de detecciones se conserva únicamente en memoria durante la sesión actual.
- El modo forense depende de la respuesta real de `POST /detect`.
- `callDetectStreamingApi` existe, pero Live Call usa WebSocket `/ws/detect`.
- El frontend no implementa el entrenamiento; inicia, consulta y cancela el proceso mediante la API.
- El progreso de entrenamiento depende de los estados persistidos por el backend.
- La normalización mono a estéreo duplica la fuente en el Canal 1; no reconstruye una pista real de agente.
- No existen tests automatizados declarados en el proyecto.

## Próximos pasos

Estas son mejoras posibles, no funcionalidades actuales:

- Añadir tests unitarios para WAV, Base64 y STFT.
- Añadir pruebas de integración para `POST /detect`, WebSocket y endpoints de revisión.
- Persistir el historial de detecciones mediante el backend.
- Añadir reconexión explícita para Live Call.
- Tipar el proyecto con TypeScript.
- Añadir configuración de despliegue para servir `dist/`.
