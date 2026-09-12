/**
 * AuraVoice Audio Utilities
 * Telephony-grade 8kHz Stereo 16-bit PCM Audio Processing & Base64 Encoder
 * Compliant with HackMTY26 Altur Challenge Specifications:
 * - Stereo WAV (Channel 0 = Caller, Channel 1 = Agent)
 * - 8,000 Hz sample rate
 * - 16-bit PCM
 * - Base64 encoding for POST /detect payload
 */

/**
 * Encodes an AudioBuffer into an 8kHz 16-bit Stereo PCM WAV ArrayBuffer
 * @param {AudioBuffer} audioBuffer 
 * @returns {ArrayBuffer}
 */
export function encodeStereoWav8kHz(audioBuffer) {
  const numChannels = 2;
  const sampleRate = 8000;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;

  const ch0 = audioBuffer.getChannelData(0);
  // Channel 1 can be either channel 1 of the buffer or an artificial agent reference
  const ch1 = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : new Float32Array(ch0.length);

  const numSamples = ch0.length;
  const dataSize = numSamples * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // --- RIFF Header ---
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // --- fmt Subchunk ---
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // --- data Subchunk ---
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Interleave Channel 0 (Caller) and Channel 1 (Agent)
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    // Channel 0 sample (Caller)
    const s0 = Math.max(-1, Math.min(1, ch0[i]));
    const val0 = s0 < 0 ? s0 * 0x8000 : s0 * 0x7FFF;
    view.setInt16(offset, val0, true);
    offset += 2;

    // Channel 1 sample (Agent)
    const s1 = Math.max(-1, Math.min(1, ch1[i]));
    const val1 = s1 < 0 ? s1 * 0x8000 : s1 * 0x7FFF;
    view.setInt16(offset, val1, true);
    offset += 2;
  }

  return buffer;
}

/**
 * Resamples an arbitrary browser AudioBuffer down to 8000 Hz Stereo
 * @param {AudioBuffer} sourceBuffer 
 * @returns {Promise<AudioBuffer>}
 */
export async function resampleTo8kHzStereo(sourceBuffer) {
  const targetSampleRate = 8000;
  const duration = sourceBuffer.duration;
  const targetLength = Math.ceil(duration * targetSampleRate);

  const offlineCtx = new OfflineAudioContext(2, targetLength, targetSampleRate);

  const bufferSource = offlineCtx.createBufferSource();
  bufferSource.buffer = sourceBuffer;

  // Route source channel 0 to offlineCtx channel 0
  const merger = offlineCtx.createChannelMerger(2);
  const splitter = offlineCtx.createChannelSplitter(sourceBuffer.numberOfChannels);

  bufferSource.connect(splitter);
  splitter.connect(merger, 0, 0); // Source Ch0 -> Target Ch0 (Caller)

  if (sourceBuffer.numberOfChannels > 1) {
    splitter.connect(merger, 1, 1); // Source Ch1 -> Target Ch1 (Agent)
  } else {
    // If mono mic recording, duplicate or generate a quiet synthetic tone/comfort noise for channel 1
    splitter.connect(merger, 0, 1);
  }

  merger.connect(offlineCtx.destination);
  bufferSource.start(0);

  return await offlineCtx.startRendering();
}

/**
 * Converts ArrayBuffer to Base64 string
 * @param {ArrayBuffer} arrayBuffer 
 * @returns {string}
 */
export function arrayBufferToBase64(arrayBuffer) {
  let binary = '';
  const bytes = new Uint8Array(arrayBuffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

/**
 * Base64 to ArrayBuffer
 * @param {string} base64 
 * @returns {ArrayBuffer}
 */
export function base64ToArrayBuffer(base64) {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/**
 * Parses the RIFF/WAVE header of a WAV ArrayBuffer without decoding the payload.
 * Returns { audioFormat, channels, sampleRate, bitsPerSample, dataSize, dataOffset }
 * or null when the buffer is not a valid RIFF/WAVE file.
 * @param {ArrayBuffer} arrayBuffer
 */
export function inspectWavMetadata(arrayBuffer) {
  try {
    if (arrayBuffer.byteLength < 44) return null;
    const view = new DataView(arrayBuffer);

    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
    if (riff !== 'RIFF' || wave !== 'WAVE') return null;

    let audioFormat = null;
    let channels = null;
    let sampleRate = null;
    let bitsPerSample = null;
    let dataSize = 0;
    let dataOffset = 0;

    let offset = 12;
    while (offset + 8 <= arrayBuffer.byteLength) {
      const id = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
      const size = view.getUint32(offset + 4, true);

      if (id === 'fmt ') {
        audioFormat = view.getUint16(offset + 8, true);
        channels = view.getUint16(offset + 10, true);
        sampleRate = view.getUint32(offset + 12, true);
        bitsPerSample = view.getUint16(offset + 22, true);
      } else if (id === 'data') {
        dataOffset = offset + 8;
        dataSize = size;
        break;
      }

      offset += 8 + size + (size % 2);
    }

    if (audioFormat == null || dataOffset === 0) return null;
    return { audioFormat, channels, sampleRate, bitsPerSample, dataSize, dataOffset };
  } catch {
    return null;
  }
}

/**
 * Reads raw interleaved PCM samples directly from a WAV ArrayBuffer (no decode).
 * Returns { channel0, channel1 } as Float32Array (normalized -1..1).
 * Mono sources are duplicated to both channels; >2 channels are reduced to the first two.
 * @param {ArrayBuffer} arrayBuffer
 * @param {object} meta A valid result from inspectWavMetadata
 */
export function readWavPcmSamples(arrayBuffer, meta) {
  const view = new DataView(arrayBuffer);
  const bytesPerSample = meta.bitsPerSample / 8;
  const frames = Math.floor(meta.dataSize / (bytesPerSample * meta.channels));

  const channel0 = new Float32Array(frames);
  const channel1 = new Float32Array(frames);

  for (let i = 0; i < frames; i++) {
    const base = meta.dataOffset + i * bytesPerSample * meta.channels;
    channel0[i] = readPcmSample(view, base, meta);
    channel1[i] = meta.channels > 1 ? readPcmSample(view, base + bytesPerSample, meta) : readPcmSample(view, base, meta);
  }

  return { channel0, channel1 };
}

function readPcmSample(view, offset, meta) {
  const bytes = meta.bitsPerSample / 8;
  if (meta.audioFormat === 3 && bytes === 4) {
    return Math.max(-1, Math.min(1, view.getFloat32(offset, true)));
  }
  if (bytes === 1) {
    return (view.getUint8(offset) - 128) / 128;
  }
  if (bytes === 2) {
    return view.getInt16(offset, true) / 32768;
  }
  if (bytes === 3) {
    let v = view.getUint8(offset) | (view.getUint8(offset + 1) << 8) | (view.getUint8(offset + 2) << 16);
    if (v & 0x800000) v -= 0x1000000;
    return v / 8388608;
  }
  if (bytes === 4) {
    return view.getInt32(offset, true) / 2147483648;
  }
  return 0;
}

/**
 * Human friendly byte size formatter (e.g. "24 KB")
 * @param {number} bytes
 */
export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// --- Spectrogram (Sliding-window STFT) --------------------------------

/**
 * In-place radix-2 FFT. Requires re/im lengths to be a power of two.
 */
function fftRadix2(re, im) {
  const n = re.length;
  let bits = 0;
  while ((1 << bits) < n) bits++;

  for (let i = 0; i < n; i++) {
    let j = 0;
    for (let b = 0; b < bits; b++) {
      j = (j << 1) | ((i >> b) & 1);
    }
    if (j > i) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = curRe * re[i + j + len / 2] - curIm * im[i + j + len / 2];
        const vIm = curRe * im[i + j + len / 2] + curIm * re[i + j + len / 2];
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

/**
 * Computes a normalized spectrogram (STFT magnitude grid) from a mono channel.
 * Returns { frames, bins, data: Float32Array (frames*bins, 0..1), sampleRate, winSize, hop, duration, maxFreq }
 * or null when the channel is too short.
 * @param {Float32Array} channelData
 * @param {number} sampleRate
 */
export function computeSpectrogram(channelData, sampleRate = 8000, opts = {}) {
  if (!channelData || channelData.length === 0) return null;

  const winSize = opts.windowSize || 256;
  const hop = opts.hop || Math.floor(winSize / 4);
  const n = channelData.length;
  if (n < winSize) return null;

  const bins = winSize / 2 + 1;
  const frames = 1 + Math.floor((n - winSize) / hop);

  const windowFn = new Float32Array(winSize);
  for (let i = 0; i < winSize; i++) {
    windowFn[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (winSize - 1));
  }

  const re = new Float32Array(winSize);
  const im = new Float32Array(winSize);
  const grid = new Float32Array(frames * bins);

  let minDb = Infinity;
  let maxDb = -Infinity;

  for (let f = 0; f < frames; f++) {
    const off = f * hop;
    for (let i = 0; i < winSize; i++) {
      re[i] = channelData[off + i] * windowFn[i];
      im[i] = 0;
    }
    fftRadix2(re, im);
    for (let b = 0; b < bins; b++) {
      const mag = Math.sqrt(re[b] * re[b] + im[b] * im[b]);
      const db = 20 * Math.log10(mag + 1e-10) - 80;
      grid[f * bins + b] = db;
      if (db < minDb) minDb = db;
      if (db > maxDb) maxDb = db;
    }
  }

  const range = Math.max(1e-6, maxDb - minDb);
  const data = new Float32Array(grid.length);
  for (let i = 0; i < grid.length; i++) {
    data[i] = Math.min(1, Math.max(0, (grid[i] - minDb) / range));
  }

  return {
    frames,
    bins,
    data,
    sampleRate,
    winSize,
    hop,
    duration: n / sampleRate,
    maxFreq: sampleRate / 2
  };
}

/**
 * Generates synthetic mock call data for test demonstrations (Human or Deepfake)
 */
export function generateMockScenario(type = 'deepfake') {
  const sampleRate = 8000;
  const durationSec = 6.0;
  const totalSamples = sampleRate * durationSec;

  const ch0 = new Float32Array(totalSamples);
  const ch1 = new Float32Array(totalSamples);

  for (let i = 0; i < totalSamples; i++) {
    const t = i / sampleRate;

    // Agent speaks during t = 0s to 2.2s and 4.2s to 5.0s
    if ((t >= 0.2 && t <= 2.2) || (t >= 4.2 && t <= 5.2)) {
      ch1[i] = Math.sin(2 * Math.PI * 220 * t) * 0.4 * (0.8 + 0.2 * Math.sin(10 * t)) * (Math.random() * 0.15 + 0.85);
    }

    // Caller speaks during t = 2.4s to 4.1s and 5.3s to 6.0s
    if ((t >= 2.4 && t <= 4.1) || (t >= 5.3 && t <= 5.9)) {
      if (type === 'deepfake') {
        // Flat pitch, unnatural harmonic frequency typical of neural vocoders (e.g. Hifi-GAN / ElevenLabs)
        const fundamental = Math.sin(2 * Math.PI * 185 * t);
        const harmonic = 0.5 * Math.sin(2 * Math.PI * 370 * t);
        const glitch = 0.2 * Math.sin(2 * Math.PI * 3800 * t); // Telephony upper band artifact
        ch0[i] = (fundamental + harmonic + glitch) * 0.5;
      } else {
        // Natural human speech: prosody variation, subtle breathing and irregular pauses
        const naturalPitch = 140 + 25 * Math.sin(3 * t);
        ch0[i] = Math.sin(2 * Math.PI * naturalPitch * t) * (0.3 + 0.3 * Math.sin(8 * t)) * (Math.random() * 0.3 + 0.7);
      }
    }
  }

  if (type === 'deepfake') {
    return {
      type: 'deepfake',
      title: 'Muestra WAV: Clonación de Voz (Deepfake Neuronal)',
      is_synthetic: true,
      confidence: 0.942,
      latency_ms: 112,
      channel0: ch0,
      channel1: ch1,
      duration: durationSec,
      metrics: {
        acoustic_score: 93,
        spectral_artifacts: 'Frecuencias anómalas en 3.8 kHz y armónicos artificialmente estables.',
        turn_recovery_ms: 180,
        conversational_messiness: 12,
        breathing_detected: false,
        semantic_hallucination: true,
      },
      llm_conclusion: `🚨 ALERTA: Voz sintética detectada con 94.2% de certeza.
Se encontró una firma acústica plana sin micro-respiraciones naturales. Al ser interrumpido por el agente, respondió con una latencia mecánica constante de 180ms. Se sugiere bloquear la llamada y pedir verificación biométrica adicional.`
    };
  } else {
    return {
      type: 'human',
      title: 'Muestra WAV: Cliente Humano Auténtico',
      is_synthetic: false,
      confidence: 0.978,
      latency_ms: 98,
      channel0: ch0,
      channel1: ch1,
      duration: durationSec,
      metrics: {
        acoustic_score: 8,
        spectral_artifacts: 'Modulación de frecuencia natural; degradación telefónica esperada.',
        turn_recovery_ms: 420,
        conversational_messiness: 88,
        breathing_detected: true,
        semantic_hallucination: false,
      },
      llm_conclusion: `✅ VERIFICADO: Voz humana auténtica confirmada (97.8% de confianza).
El tono presenta inflexiones orgánicas, respiración audible entre frases y un tiempo de vacilación natural de 420ms ante las preguntas del banco. El canal telefónico es seguro.`
    };
  }
}

/**
 * Calls the batch POST /detect endpoint.
 * Sends the raw WAV as multipart/form-data under the field `file`,
 * matching the FastAPI contract (UploadFile `file`).
 */
export async function callDetectApi(baseUrl, base64Audio) {
  const url = `${baseUrl.replace(/\/+$/, '')}/detect`;

  const fileBytes = base64ToArrayBuffer(base64Audio);
  if (!fileBytes || fileBytes.byteLength < 44) {
    // Not a real WAV payload (e.g. preloaded placeholder base64) -> demo fallback
    throw new Error('No raw WAV payload available; showing demo verdict.');
  }

  const blob = new Blob([fileBytes], { type: 'audio/wav' });
  const formData = new FormData();
  formData.append('file', blob, 'audio.wav');

  const response = await fetch(url, {
    method: 'POST',
    body: formData
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from /detect`);
  }
  return await response.json();
}

/**
 * Calls the live streaming POST /detect_streaming endpoint
 */
export async function callDetectStreamingApi(baseUrl, payload) {
  const url = `${baseUrl.replace(/\/+$/, '')}/detect_streaming`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from /detect_streaming`);
  }
  return await response.json();
}

