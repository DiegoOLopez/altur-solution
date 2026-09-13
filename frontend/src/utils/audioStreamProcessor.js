/**
 * AudioWorkletProcessor para capturar audio del micrófono.
 * 
 * Convierte los samples Float32 del navegador a enteros de 16 bits (PCM 16-bit)
 * para enviarlos a través del WebSocket de streaming de Altur.
 */
class PCMProcessor extends AudioWorkletProcessor {
  /**
   * Procesa un bloque de audio (128 samples a la vez).
   * @param {Float32Array[][]} inputs - Arrays de canales de entrada.
   * @returns {boolean} - true para mantener vivo el procesador.
   */
  process(inputs) {
    const input = inputs[0];

    // Si no hay input o canal 0, salir temprano.
    if (!input || !input[0]) {
      return true;
    }

    const samples = input[0];

    // Crear buffer Int16 para almacenar las muestras convertidas
    const pcm = new Int16Array(samples.length);

    // Convertir de Float32 [-1.0, 1.0] a PCM de 16-bit [-32768, 32767]
    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]));

      pcm[i] = sample < 0
        ? sample * 32768
        : sample * 32767;
    }

    // Enviar el buffer binario al hilo principal a través del MessagePort
    this.port.postMessage(pcm.buffer, [pcm.buffer]);

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);