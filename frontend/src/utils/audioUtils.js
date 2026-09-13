/**
 * AuraVoice - Utilidades de audio.
 *
 * Procesamiento de audio de telefonía (8 kHz, estéreo, PCM 16-bit) para el
 * reto de Tecnologías Altur:
 *
 * - WAV estéreo: Canal 0 = Llamante, Canal 1 = Agente.
 * - Sample rate telefónico de 8,000 Hz.
 * - Transmisión al backend como multipart/form-data en POST /detect_wav.
 *
 * También incluye helpers para inspeccionar WAV sin decodificarlos, generar
 * el espectrograma (STFT) y comunicarse con la API.
 */

// ---------------------------------------------------------------------------
// Codificador WAV (8 kHz, estéreo, PCM 16-bit)
// ---------------------------------------------------------------------------

function writeString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

/**
 * Convierte un AudioBuffer del navegador a un ArrayBuffer de WAV 
 * con formato 8kHz 16-bit PCM Estéreo.
 * 
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
  // El canal 1 puede ser del buffer o un silencio artificial para el agente.
  const ch1 = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : new Float32Array(ch0.length);

  const numSamples = ch0.length;
  const dataSize = numSamples * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // --- Cabecera RIFF ---
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // --- Subchunk fmt ---
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 para PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 para PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // --- Subchunk data ---
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Intercalar Canal 0 (Llamante) y Canal 1 (Agente)
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    // Muestra del Canal 0
    const s0 = Math.max(-1, Math.min(1, ch0[i]));
    const val0 = s0 < 0 ? s0 * 0x8000 : s0 * 0x7FFF;
    view.setInt16(offset, val0, true);
    offset += 2;

    // Muestra del Canal 1
    const s1 = Math.max(-1, Math.min(1, ch1[i]));
    const val1 = s1 < 0 ? s1 * 0x8000 : s1 * 0x7FFF;
    view.setInt16(offset, val1, true);
    offset += 2;
  }

  return buffer;
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/**
 * Hace resample de un AudioBuffer genérico del navegador a 8000 Hz Estéreo.
 * Utiliza OfflineAudioContext para el procesamiento.
 * 
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

  const merger = offlineCtx.createChannelMerger(2);
  const splitter = offlineCtx.createChannelSplitter(sourceBuffer.numberOfChannels);

  bufferSource.connect(splitter);
  splitter.connect(merger, 0, 0); // Ch0 a Ch0

  if (sourceBuffer.numberOfChannels > 1) {
    splitter.connect(merger, 1, 1); // Ch1 a Ch1
  } else {
    // Duplicar canal si es mono
    splitter.connect(merger, 0, 1);
  }

  merger.connect(offlineCtx.destination);
  bufferSource.start(0);

  return await offlineCtx.startRendering();
}

// ---------------------------------------------------------------------------
// Parsing de metadatos WAV
// ---------------------------------------------------------------------------

/**
 * Lee la cabecera RIFF/WAVE sin decodificar el contenido.
 * Devuelve metadata o null si es inválido.
 * 
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
      const id = String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3)
      );
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
 * Lee muestras PCM directamente del ArrayBuffer sin decodificar el WAV entero.
 * Devuelve { channel0, channel1 } normalizados (-1 a 1).
 * 
 * @param {ArrayBuffer} arrayBuffer
 * @param {object} meta Metadata generada por inspectWavMetadata
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
    channel1[i] =
      meta.channels > 1
        ? readPcmSample(view, base + bytesPerSample, meta)
        : readPcmSample(view, base, meta);
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
    const v =
      view.getUint8(offset) |
      (view.getUint8(offset + 1) << 8) |
      (view.getUint8(offset + 2) << 16);
    return (v & 0x800000 ? v - 0x1000000 : v) / 8388608;
  }
  if (bytes === 4) {
    return view.getInt32(offset, true) / 2147483648;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Conversión Base64
// ---------------------------------------------------------------------------

/**
 * Convierte ArrayBuffer a Base64
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
 * Convierte Base64 a ArrayBuffer
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

// ---------------------------------------------------------------------------
// Formateo para la UI
// ---------------------------------------------------------------------------

/**
 * Formatea un tamaño de bytes para legibilidad humana.
 */
export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ---------------------------------------------------------------------------
// Espectrograma
// ---------------------------------------------------------------------------

/**
 * Transformada rápida de Fourier (FFT) in-place.
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
 * Calcula un espectrograma (STFT) normalizado.
 * Genera una matriz unidimensional visualizable en canvas.
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

// ---------------------------------------------------------------------------
// Simulación de Escenarios (Demos UI)
// ---------------------------------------------------------------------------

/**
 * Genera datos de llamada de simulación (Humano o Deepfake) para UI demos.
 */
export function generateMockScenario(type = 'deepfake') {
  const sampleRate = 8000;
  const durationSec = 6.0;
  const totalSamples = sampleRate * durationSec;

  const ch0 = new Float32Array(totalSamples);
  const ch1 = new Float32Array(totalSamples);

  for (let i = 0; i < totalSamples; i++) {
    const t = i / sampleRate;

    if ((t >= 0.2 && t <= 2.2) || (t >= 4.2 && t <= 5.2)) {
      ch1[i] = Math.sin(2 * Math.PI * 220 * t) * 0.4 * (0.8 + 0.2 * Math.sin(10 * t)) * (Math.random() * 0.15 + 0.85);
    }

    if ((t >= 2.4 && t <= 4.1) || (t >= 5.3 && t <= 5.9)) {
      if (type === 'deepfake') {
        const fundamental = Math.sin(2 * Math.PI * 185 * t);
        const harmonic = 0.5 * Math.sin(2 * Math.PI * 370 * t);
        const glitch = 0.2 * Math.sin(2 * Math.PI * 3800 * t);
        ch0[i] = (fundamental + harmonic + glitch) * 0.5;
      } else {
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
      llm_conclusion: `ALERTA: Voz sintética detectada con 94.2% de certeza.
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
      llm_conclusion: `VERIFICADO: Voz humana auténtica confirmada (97.8% de confianza).
El tono presenta inflexiones orgánicas, respiración audible entre frases y un tiempo de vacilación natural de 420ms ante las preguntas del banco. El canal telefónico es seguro.`
    };
  }
}

// ---------------------------------------------------------------------------
// Peticiones al API
// ---------------------------------------------------------------------------

/**
 * Envía un archivo WAV a POST /detect_wav como multipart/form-data.
 */
export async function callDetectApi(baseUrl, base64Audio) {
  const url = `${baseUrl.replace(/\/+$/, '')}/detect_wav`;

  const fileBytes = base64ToArrayBuffer(base64Audio);
  if (!fileBytes || fileBytes.byteLength < 44) {
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
    throw new Error(`HTTP ${response.status} from /detect_wav`);
  }
  return await response.json();
}