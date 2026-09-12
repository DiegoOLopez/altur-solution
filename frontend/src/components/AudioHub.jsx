import React, { useState, useRef, useEffect } from 'react';
import { 
  Mic, Square, Upload, Play, RefreshCw, FileAudio, 
  CheckCircle2, AlertCircle, Sparkles, Binary, Zap 
} from 'lucide-react';
import { 
  encodeStereoWav8kHz, 
  resampleTo8kHzStereo, 
  arrayBufferToBase64,
  generateMockScenario 
} from '../utils/audioUtils';

export default function AudioHub({ 
  onAnalysisTrigger, 
  isAnalyzing, 
  onAudioLoaded,
  activeAudioInfo 
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [inputMode, setInputMode] = useState('presets'); // 'presets' | 'mic' | 'file'
  const [base64Preview, setBase64Preview] = useState('');
  const [audioSourceDesc, setAudioSourceDesc] = useState('Ningún audio seleccionado');

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerIntervalRef = useRef(null);
  const fileInputRef = useRef(null);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  // Pre-load default deepfake mock on initial mount so the UI starts with lively data
  useEffect(() => {
    handleSelectPreset('deepfake');
  }, []);

  // Handler for Quick Test Presets
  const handleSelectPreset = (type) => {
    const mock = generateMockScenario(type);
    setAudioSourceDesc(mock.title);
    
    // Create an AudioBuffer to encode to Base64
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioCtx.createBuffer(2, mock.channel0.length, 8000);
    audioBuffer.copyToChannel(mock.channel0, 0);
    audioBuffer.copyToChannel(mock.channel1, 1);

    const wavArrayBuffer = encodeStereoWav8kHz(audioBuffer);
    const b64 = arrayBufferToBase64(wavArrayBuffer);
    setBase64Preview(b64);

    onAudioLoaded({
      title: mock.title,
      channel0: mock.channel0,
      channel1: mock.channel1,
      duration: mock.duration,
      base64: b64,
      mockResult: mock
    });
  };

  // Start Live Microphone Recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        // Resample to 8kHz Stereo (Channel 0 = Caller mic, Channel 1 = Reference)
        const resampledBuffer = await resampleTo8kHzStereo(decodedBuffer);
        const wavArrayBuffer = encodeStereoWav8kHz(resampledBuffer);
        const b64 = arrayBufferToBase64(wavArrayBuffer);

        setBase64Preview(b64);
        setAudioSourceDesc(`Grabación de Micrófono (${resampledBuffer.duration.toFixed(1)}s)`);

        onAudioLoaded({
          title: `Grabación en Vivo (${resampledBuffer.duration.toFixed(1)}s)`,
          channel0: resampledBuffer.getChannelData(0),
          channel1: resampledBuffer.getChannelData(1),
          duration: resampledBuffer.duration,
          base64: b64,
          mockResult: null
        });

        // Stop all mic tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordDuration(0);

      const startTime = Date.now();
      timerIntervalRef.current = setInterval(() => {
        setRecordDuration((Date.now() - startTime) / 1000);
      }, 100);

    } catch (err) {
      console.error('Error al acceder al micrófono:', err);
      alert('No se pudo acceder al micrófono. Por favor concede permisos o usa los casos de prueba.');
    }
  };

  // Stop Live Recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    }
  };

  // Handle WAV File Upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      const resampledBuffer = await resampleTo8kHzStereo(decodedBuffer);
      const wavArrayBuffer = encodeStereoWav8kHz(resampledBuffer);
      const b64 = arrayBufferToBase64(wavArrayBuffer);

      setBase64Preview(b64);
      setAudioSourceDesc(`${file.name} (${resampledBuffer.duration.toFixed(1)}s)`);

      onAudioLoaded({
        title: file.name,
        channel0: resampledBuffer.getChannelData(0),
        channel1: resampledBuffer.getChannelData(1),
        duration: resampledBuffer.duration,
        base64: b64,
        mockResult: null
      });
    } catch (err) {
      console.error('Error al procesar archivo de audio:', err);
      alert('Error al leer el archivo de audio. Asegúrate de que sea un formato de audio válido.');
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header & Source Mode Tabs */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileAudio size={18} color="#00f2fe" />
          <h2 style={{ fontSize: '1rem', fontWeight: '700', letterSpacing: '-0.01em' }}>
            1. Entrada de Audio Telefónico
          </h2>
        </div>
        <span className="chip chip-cyan" style={{ fontSize: '0.65rem' }}>
          Formato Fase 1
        </span>
      </div>

      {/* Input Selector Tabs */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '6px',
        background: 'var(--bg-tertiary)',
        padding: '4px',
        borderRadius: 'var(--radius-sm)'
      }}>
        <button
          className={`btn ${inputMode === 'presets' ? 'btn-primary' : 'btn-outline'}`}
          style={{ padding: '8px 6px', fontSize: '0.78rem' }}
          onClick={() => setInputMode('presets')}
        >
          <Sparkles size={14} />
          Casos Demo
        </button>
        <button
          className={`btn ${inputMode === 'mic' ? 'btn-primary' : 'btn-outline'}`}
          style={{ padding: '8px 6px', fontSize: '0.78rem' }}
          onClick={() => setInputMode('mic')}
        >
          <Mic size={14} />
          Micrófono
        </button>
        <button
          className={`btn ${inputMode === 'file' ? 'btn-primary' : 'btn-outline'}`}
          style={{ padding: '8px 6px', fontSize: '0.78rem' }}
          onClick={() => setInputMode('file')}
        >
          <Upload size={14} />
          Cargar WAV
        </button>
      </div>

      {/* Dynamic Tab Body */}
      {inputMode === 'presets' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Selecciona muestras sintéticas o humanas para simulación rápida:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <button
              className="btn btn-outline"
              style={{
                borderColor: 'rgba(239, 68, 68, 0.4)',
                background: 'rgba(239, 68, 68, 0.08)',
                padding: '12px',
                flexDirection: 'column',
                alignItems: 'flex-start',
                textAlign: 'left'
              }}
              onClick={() => handleSelectPreset('deepfake')}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%', justifyContent: 'space-between' }}>
                <span style={{ color: '#ef4444', fontWeight: 700, fontSize: '0.82rem' }}>🔴 Deepfake TTS</span>
                <span className="chip chip-rose" style={{ fontSize: '0.6rem' }}>Sintético</span>
              </div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Voz generada por IA con artefactos vocoder
              </span>
            </button>

            <button
              className="btn btn-outline"
              style={{
                borderColor: 'rgba(16, 185, 129, 0.4)',
                background: 'rgba(16, 185, 129, 0.08)',
                padding: '12px',
                flexDirection: 'column',
                alignItems: 'flex-start',
                textAlign: 'left'
              }}
              onClick={() => handleSelectPreset('human')}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%', justifyContent: 'space-between' }}>
                <span style={{ color: '#10b981', fontWeight: 700, fontSize: '0.82rem' }}>🟢 Humano Legítimo</span>
                <span className="chip chip-emerald" style={{ fontSize: '0.6rem' }}>Real</span>
              </div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Voz humana con prosodia natural y pausas
              </span>
            </button>
          </div>
        </div>
      )}

      {inputMode === 'mic' && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          padding: '16px',
          background: 'var(--bg-tertiary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px dashed var(--border-subtle)'
        }}>
          {!isRecording ? (
            <button
              className="btn btn-primary"
              style={{ padding: '12px 24px', fontSize: '0.95rem' }}
              onClick={startRecording}
            >
              <Mic size={18} />
              Iniciar Grabación de Llamada
            </button>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="pulse-dot rose"></span>
                <span style={{ color: '#ef4444', fontWeight: 700, fontSize: '0.9rem' }}>
                  GRABANDO AUDIO EN VIVO
                </span>
              </div>
              <span className="font-mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: '#00f2fe' }}>
                {recordDuration.toFixed(1)}s
              </span>
              <button
                className="btn btn-danger"
                style={{ padding: '10px 20px' }}
                onClick={stopRecording}
              >
                <Square size={16} fill="#fff" />
                Detener y Procesar (8kHz Estéreo)
              </button>
            </div>
          )}
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center' }}>
            El audio del micrófono se asigna a Canal 0 (Caller) y se resamplea a 8kHz 16-bit PCM.
          </span>
        </div>
      )}

      {inputMode === 'file' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <input
            type="file"
            accept=".wav,.mp3,.ogg,.flac"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button
            className="btn btn-outline"
            style={{
              padding: '24px',
              flexDirection: 'column',
              gap: '10px',
              borderStyle: 'dashed'
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={24} color="#00f2fe" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
              Arrastra o haz clic para cargar WAV de Altur
            </span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Acepta archivos de audio estéreo o mono telefónico
            </span>
          </button>
        </div>
      )}

      {/* Active Audio Metadata & Base64 Payload Inspector */}
      <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-sm)',
        padding: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
            Audio Activo:
          </span>
          <span style={{ fontSize: '0.75rem', color: '#00f2fe', fontWeight: 600 }}>
            {audioSourceDesc}
          </span>
        </div>

        {/* Base64 Badge info */}
        {base64Preview && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.72rem',
            color: 'var(--text-muted)',
            paddingTop: '6px',
            borderTop: '1px solid rgba(255,255,255,0.05)'
          }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Binary size={12} color="#10b981" />
              Base64 listo ({Math.round(base64Preview.length / 1024)} KB)
            </span>
            <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>
              {base64Preview.substring(0, 16)}...
            </span>
          </div>
        )}
      </div>

      {/* Primary Trigger Button */}
      <button
        className="btn btn-primary"
        style={{
          width: '100%',
          padding: '14px',
          fontSize: '0.98rem',
          fontWeight: 700,
          letterSpacing: '0.02em'
        }}
        onClick={onAnalysisTrigger}
        disabled={isAnalyzing || isRecording}
      >
        {isAnalyzing ? (
          <>
            <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
            Enviando a POST /detect y Analizando...
          </>
        ) : (
          <>
            <Zap size={18} fill="#04101e" />
            EJECUTAR DETECCIÓN FORENSE
          </>
        )}
      </button>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
