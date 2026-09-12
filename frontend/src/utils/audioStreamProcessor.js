class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const samples = input[0];

    const pcm = new Int16Array(samples.length);

    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]));

      pcm[i] = sample < 0
        ? sample * 32768
        : sample * 32767;
    }

    this.port.postMessage(pcm.buffer, [pcm.buffer]);

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);