# 1. Flujo de entrada

_Front-end / UI._

## Flujo de datos

1. La UI recibe el audio.
2. El audio se representa como información **estéreo, 8 kHz y 16-bit PCM**.
3. Se codifica el fragmento de audio en **Base64**.
4. Se llama a la API enviando el audio en el payload de la petición HTTP.

## Procesamiento

La petición viaja a los motores de análisis y permanece esperando hasta recibir una respuesta.

> **Nota de diseño:** El valor `64` indica que el fragmento de audio se codificará en Base64 para enviarlo en el payload de la petición HTTP.
