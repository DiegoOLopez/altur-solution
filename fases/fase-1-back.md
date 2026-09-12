# Arquitectura Eureka: Fase 1

## Backend MVP dual con FastAPI

### Objetivo

Levantar el motor central de análisis mediante dos interfaces de comunicación independientes:

- Un endpoint HTTP para cumplir con el formato de evaluación automatizada del reto Altur.
- Un canal de streaming de muy baja latencia para la demostración en vivo.

## 1. Endpoint oficial

Este endpoint está diseñado exclusivamente para recibir las peticiones del script evaluador de HackMTY.

| Propiedad | Especificación |
| --- | --- |
| Método y ruta | `POST /detect` |
| Entrada | Clip WAV codificado en `Base64` dentro del payload de la petición |
| Formato de origen | 8 kHz, 16-bit PCM, estéreo (2 canales) |

### Mapeo de canales

- `Channel 0`: llamador, cuya voz será analizada.
- `Channel 1`: agente de IA bancario, utilizado como contexto temporal.

### Salida esperada

La respuesta debe cumplir estrictamente con el siguiente formato JSON:

```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
```

## 2. Endpoint de streaming

Este endpoint se utilizará en la interfaz gráfica del equipo para simular un entorno de Contact Center (CCaaS) en tiempo real.

| Propiedad | Especificación |
| --- | --- |
| Protocolo y ruta | `WebSocket /ws/detect` |
| Comunicación | Canal bidireccional continuo |
| Entrada | Flujo continuo de fragmentos binarios crudos (`Raw PCM`, mono) |
| Salida | JSON emitido periódicamente con scores desglosados para actualizar las gráficas del front-end |

## 3. Tubería de datos y estrategia de procesamiento

Para garantizar la compatibilidad con los modelos modernos de IA sin sacrificar precisión, el backend ejecutará los siguientes pasos secuenciales:

### 3.1 Ingesta y decodificación

Extraer el audio, ya sea desde `Base64` o desde el stream binario, y cargarlo en la memoria del servidor.

### 3.2 Separación espacial

Aplica únicamente al endpoint `POST /detect`:

- Aislar `Channel 0` para el análisis acústico.
- Conservar `Channel 1` para medir interrupciones y latencias.

### 3.3 Conversión de frecuencia

Transformar internamente el audio de 8 kHz a 16 kHz. Este paso es obligatorio para que los modelos preentrenados de análisis acústico y detección de spoofing no produzcan errores de dimensionalidad ni falsos positivos.

### 3.4 Análisis multicapa

- **Análisis acústico:** analizar `Channel 0`, ahora en 16 kHz, en busca de firmas sintéticas.
- **Análisis comportamental:** medir los tiempos de respuesta cruzando `Channel 0` y `Channel 1`.
  - Esta métrica tendrá un peso mayor para compensar la falta de frecuencias altas reales producida por el upsampling.

### 3.5 Veredicto

Ponderar los resultados de los análisis y emitir la respuesta final.
