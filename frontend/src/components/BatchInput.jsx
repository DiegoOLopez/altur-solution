import React, { useRef, useState, useEffect } from 'react';
import { UploadCloud, Sparkles, FileAudio, Bot, UserCheck, ArrowRight } from 'lucide-react';
import {
  encodeStereoWav8kHz,
  resampleTo8kHzStereo,
  arrayBufferToBase64,
  generateMockScenario,
  inspectWavMetadata,
  readWavPcmSamples,
  formatBytes
} from '../utils/audioUtils';
import ForensicPipeline from './ForensicPipeline';

const INGESTION_STEPS = [
  { id: 'received', label: 'Audio recibido', sublabel: () => 'Señal entrante registrada en el canal' },
  { id: 'format', label: 'Validación de formato', sublabel: (d) => d?.format || 'En espera de archivo' },
  { id: 'stereo', label: 'Validación estéreo', sublabel: (d) => d?.stereo || 'En espera de archivo' },
  { id: 'rate', label: 'Validación 8 kHz', sublabel: (d) => d?.rate || 'En espera de archivo' },
  { id: 'pcm', label: 'Validación 16-bit PCM', sublabel: (d) => d?.pcm || 'En espera de archivo' },
  { id: 'split', label: 'Separación de canales', sublabel: () => 'Canal 0 (Llamante) • Canal 1 (Agente)' },
  { id: 'caller', label: 'Extracción del llamante', sublabel: () => 'Canal 0 seleccionado para clasificación' },
  { id: 'b64', label: 'Codificación Base64', sublabel: (d) => d?.base64Bytes ? `Payload ${d.base64Bytes} listo para HTTP` : 'Payload listo para HTTP' }
];

const ANALYSIS_STEPS = [
  { id: 'post', label: 'POST /detect', sublabel: () => 'HTTP 200 • application/json' },
  { id: 'ai', label: 'Análisis IA', sublabel: () => 'Motor acústico-comportamental + semántico' },
  { id: 'verdict', label: 'Veredicto forense', sublabel: () => 'Humano vs. Sintético' }
];

const PRESET_DETAIL = {
  format: 'WAV PCM nativo (dataset Altur)',
  stereo: 'Estéreo 2 canales',
  rate: '8,000 Hz exacto',
  pcm: '16-bit PCM sin pérdida',
  base64Bytes: ''
};

function buildNormDetail(meta, b64Len) {
  return {
    format: 'WAV PCM verificado (RIFF/WAVE nativo)',
    stereo: `Estéreo ${meta.channels} canales nativo`,
    rate: '8,000 Hz exacto (sin resample)',
    pcm: `16-bit PCM nativo (${meta.bitsPerSample}-bit)`,
    base64Bytes: formatBytes(b64Len)
  };
}

function buildConvertedDetail(meta, decodedBuffer, b64Len) {
  const stereoLabel = decodedBuffer.numberOfChannels === 1
    ? 'Mono → estéreo (Caller = fuente, Agente = eco)'
    : `Estéreo ${meta.channels} → 2 canales`;
  return {
    format: `WAV PCM detectado (${meta.sampleRate} Hz / ${meta.bitsPerSample}-bit)`,
    stereo: stereoLabel,
    rate: `Resampleado ${meta.sampleRate.toLocaleString('es')} Hz → 8,000 Hz`,
    pcm: `Recodificado a 16-bit PCM (${meta.bitsPerSample}-bit → 16)`,
    base64Bytes: formatBytes(b64Len)
  };
}

function buildTranscodedDetail(fileName, decodedBuffer, b64Len) {
  const ext = (fileName.split('.').pop() || 'audio').toUpperCase();
  const stereoLabel = decodedBuffer.numberOfChannels === 1
    ? 'Mono → estéreo (Caller = fuente, Agente = eco)'
    : `Estéreo ${decodedBuffer.numberOfChannels} canales`;
  return {
    format: `${ext} → transcodificado a WAV PCM`,
    stereo: stereoLabel,
    rate: `Resampleado ${Math.round(decodedBuffer.sampleRate).toLocaleString('es')} Hz → 8,000 Hz`,
    pcm: 'Codificado a 16-bit PCM',
    base64Bytes: formatBytes(b64Len)
  };
}

export default function BatchInput({
  onAudioReady,
  onRunBatchAnalysis,
  isAnalyzing,
  currentAudioTitle,
  verdictTitle
}) {
  const fileInputRef = useRef(null);
  const [detail, setDetail] = useState(null);
  const [ingestRevealed, setIngestRevealed] = useState(0);
  const [cascadeDone, setCascadeDone] = useState(false);
  const [analysisPhase, setAnalysisPhase] = useState(0);

  const verdictMatched = Boolean(verdictTitle && currentAudioTitle && verdictTitle === currentAudioTitle);

  // Cascade reveal of the ingestion steps whenever a new audio is loaded
  useEffect(() => {
    setIngestRevealed(0);
    setCascadeDone(false);

    let i = 0;
    const interval = setInterval(() => {
      i += 1;
      setIngestRevealed(i);
      if (i >= INGESTION_STEPS.length) {
        clearInterval(interval);
        setCascadeDone(true);
      }
    }, 120);

    return () => clearInterval(interval);
  }, [currentAudioTitle]);

  // Progressive status while POST /detect is analyzing
  useEffect(() => {
    if (!isAnalyzing) return;
    setAnalysisPhase(0);
    const t1 = setTimeout(() => setAnalysisPhase(1), 600);
    const t2 = setTimeout(() => setAnalysisPhase(2), 1500);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [isAnalyzing]);

  const commitAudio = (title, detailInfo, channel0, channel1, duration, base64, mockResult) => {
    setDetail(detailInfo);
    onAudioReady({ title, channel0, channel1, duration, base64, mockResult });
  };

  const handleSelectPreset = (type) => {
    const mock = generateMockScenario(type);

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioCtx.createBuffer(2, mock.channel0.length, 8000);
    audioBuffer.copyToChannel(mock.channel0, 0);
    audioBuffer.copyToChannel(mock.channel1, 1);

    const wavArrayBuffer = encodeStereoWav8kHz(audioBuffer);
    const b64 = arrayBufferToBase64(wavArrayBuffer);

    const presetDetail = { ...PRESET_DETAIL, base64Bytes: formatBytes(b64.length) };
    commitAudio(mock.title, presetDetail, mock.channel0, mock.channel1, mock.duration, b64, mock);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const arrayBuffer = await file.arrayBuffer();
      const meta = inspectWavMetadata(arrayBuffer);

      // Norm-compliant WAV (stereo • 8 kHz • 16-bit PCM): no transcoding needed
      if (meta && meta.audioFormat === 1 &&
          meta.channels === 2 && meta.sampleRate === 8000 && meta.bitsPerSample === 16) {
        const { channel0, channel1 } = readWavPcmSamples(arrayBuffer, meta);
        const b64 = arrayBufferToBase64(arrayBuffer);
        commitAudio(
          file.name,
          buildNormDetail(meta, b64.length),
          channel0,
          channel1,
          channel0.length / 8000,
          b64,
          null
        );
        return;
      }

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const resampledBuffer = await resampleTo8kHzStereo(decodedBuffer);
      const wavArrayBuffer = encodeStereoWav8kHz(resampledBuffer);
      const b64 = arrayBufferToBase64(wavArrayBuffer);

      const detailInfo = meta
        ? buildConvertedDetail(meta, decodedBuffer, b64.length)
        : buildTranscodedDetail(file.name, decodedBuffer, b64.length);

      commitAudio(
        file.name,
        detailInfo,
        resampledBuffer.getChannelData(0),
        resampledBuffer.getChannelData(1),
        resampledBuffer.duration,
        b64,
        null
      );
    } catch (err) {
      console.error('Error al procesar archivo:', err);
      alert('Formato de audio no reconocido. Por favor sube un archivo de audio válido.');
    }
  };

  const hasAudio = Boolean(currentAudioTitle);

  const ingestionSteps = INGESTION_STEPS.map((s, idx) => {
    let status = 'pending';
    if (!hasAudio) {
      status = 'pending';
    } else if (idx < ingestRevealed) {
      status = 'done';
    } else if (idx === ingestRevealed && !cascadeDone) {
      status = 'active';
    }
    return { ...s, status, sublabel: s.sublabel(detail) };
  });

  const analysisSteps = ANALYSIS_STEPS.map((s, idx) => {
    let status = 'pending';
    if (verdictMatched) {
      status = 'done';
    } else if (isAnalyzing) {
      if (idx === 0) status = analysisPhase >= 1 ? 'done' : 'active';
      else if (idx === 1) status = analysisPhase >= 2 ? 'done' : (analysisPhase >= 1 ? 'active' : 'pending');
      else status = 'pending';
    }
    return { ...s, status, sublabel: s.sublabel() };
  });

  return (
    <div className="avant-card" style={{ padding: '28px', display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Title & Description */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'var(--accent-cyan-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <FileAudio size={18} color="#0284c7" />
            </div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--altur-black)', letterSpacing: '-0.02em' }}>
              Audio Forensics
            </h2>
          </div>
          <span className="altur-badge badge-cyan font-mono">
            POST /detect
          </span>
        </div>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          Audita grabaciones telefónicas estéreo de 8kHz del dataset de Altur: el pipeline valida formato,
          separa canales y extrae al llamante antes del envío forense al endpoint.
        </p>
      </div>

      {/* Upload Dropzone */}
      <input
        type="file"
        accept=".wav,.mp3,.ogg,.flac"
        ref={fileInputRef}
        onChange={handleFileUpload}
        style={{ display: 'none' }}
      />
      <div
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: '2px dashed #cbd5e1',
          borderRadius: 'var(--radius-md)',
          background: '#f8fafc',
          padding: '28px 20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          cursor: 'pointer',
          transition: 'all 0.25s ease'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = '#0284c7';
          e.currentTarget.style.background = 'var(--accent-cyan-light)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = '#cbd5e1';
          e.currentTarget.style.background = '#f8fafc';
        }}
      >
        <div style={{
          width: '56px',
          height: '56px',
          borderRadius: '16px',
          background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: 'var(--shadow-glow-cyan)'
        }}>
          <UploadCloud size={26} color="#ffffff" />
        </div>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--altur-black)' }}>
            Arrastra tu archivo WAV o haz clic aquí
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Norma Altur: Audio estéreo 8kHz • 16-bit PCM Base64
          </p>
        </div>
      </div>

      {/* Quick Test Samples */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Sparkles size={14} color="#0284c7" />
          Muestras precargadas del dataset Altur:
        </span>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <button
            className="btn-altur-outline"
            style={{
              padding: '14px',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              textAlign: 'left'
            }}
            onClick={() => handleSelectPreset('deepfake')}
          >
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'var(--accent-rose-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Bot size={18} color="#e11d48" />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '0.86rem', color: 'var(--altur-black)' }}>Muestra Deepfake</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Síntesis neuronal TTS</div>
            </div>
          </button>

          <button
            className="btn-altur-outline"
            style={{
              padding: '14px',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              textAlign: 'left'
            }}
            onClick={() => handleSelectPreset('human')}
          >
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'var(--accent-emerald-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <UserCheck size={18} color="#059669" />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '0.86rem', color: 'var(--altur-black)' }}>Muestra Humana</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Cliente bancario real</div>
            </div>
          </button>
        </div>
      </div>

      {/* Active File Notification */}
      {hasAudio && (
        <div style={{
          background: 'var(--bg-subtle)',
          borderRadius: 'var(--radius-sm)',
          padding: '10px 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.82rem',
          gap: '10px',
          flexWrap: 'wrap'
        }}>
          <span style={{ color: 'var(--text-secondary)' }}>Archivo en análisis:</span>
          <span style={{ fontWeight: 700, color: 'var(--altur-black)', fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
            {currentAudioTitle}
          </span>
        </div>
      )}

      {/* Forensic Pipeline: Ingest & Validate */}
      <ForensicPipeline
        title="Ingesta y Validación Forense"
        badge={hasAudio ? 'WAV LISTO' : 'EN ESPERA'}
        live={hasAudio && !isAnalyzing}
        steps={ingestionSteps}
      />

      {/* Forensic Pipeline: Inference & Verdict */}
      <ForensicPipeline
        title="Inferencia y Veredicto"
        badge={isAnalyzing ? 'PROCESANDO' : 'POST /detect'}
        live={isAnalyzing}
        steps={analysisSteps}
      />

      {/* Main Trigger Action */}
      <button
        className="btn-altur btn-altur-primary"
        style={{ width: '100%', padding: '16px', fontSize: '1rem' }}
        onClick={onRunBatchAnalysis}
        disabled={isAnalyzing || !hasAudio}
      >
        {isAnalyzing ? (
          <><span className="rotate-spin" style={{ width: '15px', height: '15px', border: '2px solid rgba(255,255,255,0.35)', borderTopColor: '#ffffff', borderRadius: '50%', display: 'inline-block' }}></span>
          <span>Ejecutando POST /detect...</span></>
        ) : (
          <>
            <span>Ejecutar Detección Forense (POST /detect)</span>
            <ArrowRight size={18} />
          </>
        )}
      </button>
    </div>
  );
}