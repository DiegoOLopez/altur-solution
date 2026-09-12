import React, { useState, useEffect, useRef } from 'react';
import { Phone, PhoneOff, Mic, Volume2, Radio, UserX, UserCheck, MicOff } from 'lucide-react';
import { callDetectStreamingApi, resampleTo8kHzStereo, encodeStereoWav8kHz, arrayBufferToBase64 } from '../utils/audioUtils';

export default function StreamingCallSimulator({
  onStreamingUpdate,
  onCallFinished,
  onRecordingResult,
  apiBaseUrl = 'http://localhost:8000'
}) {
  const [isCallActive, setIsCallActive] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [streamChunksSent, setStreamChunksSent] = useState(0);
  const [currentSpeaker, setCurrentSpeaker] = useState('agent'); // 'agent' | 'caller'
  const [liveRiskScore, setLiveRiskScore] = useState(15);
  const [micLevel, setMicLevel] = useState(0);
  const [micStatus, setMicStatus] = useState('idle'); // 'idle' | 'live' | 'fallback'
  const [callSimulationType, setCallSimulationType] = useState('deepfake');

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const rAFRef = useRef(null);
  const chunkBlobsRef = useRef([]);
  const allChunksRef = useRef([]);
  const isSendingRef = useRef(false);
  const finalRiskRef = useRef(0);
  const startTimeRef = useRef(0);
  const micStatusRef = useRef('idle');

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (rAFRef.current) cancelAnimationFrame(rAFRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current) audioCtxRef.current.close().catch(() => {});
    };
  }, []);

  const cleanUpMedia = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (rAFRef.current) cancelAnimationFrame(rAFRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch { /* recorder already stopped */ }
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    micStatusRef.current = 'idle';
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  };

  const sweepMicMeter = () => {
    const analyser = audioCtxRef.current?.analyser;
    if (!analyser) return;
    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / data.length);
    setMicLevel(Math.min(1, rms * 6));
    setCurrentSpeaker(rms > 0.012 ? 'caller' : 'agent');
    rAFRef.current = requestAnimationFrame(sweepMicMeter);
  };

  const startCall = async () => {
    setIsCallActive(true);
    setCallDuration(0);
    setStreamChunksSent(0);
    setCurrentSpeaker('agent');
    setMicLevel(0);

    const isDeepfake = callSimulationType === 'deepfake';
    const startRisk = isDeepfake ? 45 : 12;
    setLiveRiskScore(startRisk);
    finalRiskRef.current = startRisk;

    const startTime = Date.now();
    startTimeRef.current = startTime;

    // Try to acquire the real microphone for a true live call
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false }
      });
      streamRef.current = stream;
      chunkBlobsRef.current = [];
      allChunksRef.current = [];
      micStatusRef.current = 'live';
      setMicStatus('live');

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunkBlobsRef.current.push(e.data);
          allChunksRef.current.push(e.data);
        }
      };
      recorder.start(400);

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioCtxRef.current = { ctx: audioCtx, analyser, audioCtxSource: source };
      rAFRef.current = requestAnimationFrame(sweepMicMeter);
    } catch (err) {
      console.warn('Micrófono no disponible, usando simulación:', err);
      micStatusRef.current = 'fallback';
      setMicStatus('fallback');
    }

    timerRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setCallDuration(elapsed);
      setStreamChunksSent((prev) => prev + 1);

      if (micStatusRef.current !== 'live') {
        if (elapsed % 6 < 3) {
          setCurrentSpeaker('agent');
        } else {
          setCurrentSpeaker('caller');
        }
      }

      if (micStatusRef.current === 'live' && !isSendingRef.current) {
        dispatchMicChunk();
      } else {
        const newRisk = isDeepfake
          ? Math.min(96, Math.floor(45 + elapsed * 8))
          : Math.max(4, Math.floor(25 - elapsed * 3));
        finalRiskRef.current = newRisk;
        setLiveRiskScore(newRisk);
        onStreamingUpdate({
          elapsed,
          chunkNumber: elapsed + 1,
          isSynthetic: isDeepfake,
          currentRisk: newRisk
        });
      }
    }, 1000);
  };

  const dispatchMicChunk = async () => {
    if (chunkBlobsRef.current.length === 0) return;
    if (micStatusRef.current !== 'live') return;
    isSendingRef.current = true;

    const chunkBlobs = chunkBlobsRef.current.slice();
    chunkBlobsRef.current = [];
    const blob = new Blob(chunkBlobs, { type: 'audio/webm' });

    try {
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await audioCtx.decodeAudioData(arrayBuffer);
      const resampled = await resampleTo8kHzStereo(decoded);
      const wav = encodeStereoWav8kHz(resampled);
      const b64 = arrayBufferToBase64(wav);

      const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);

      try {
        const data = await callDetectStreamingApi(apiBaseUrl, {
          audio: b64,
          chunk_index: elapsed,
          sample_rate: 8000
        });
        const confidence = Math.min(1, Math.max(0, Number(data?.confidence ?? 0.5)));
        const synth = Boolean(data?.is_synthetic);
        const risk = synth
          ? Math.min(98, Math.round(50 + confidence * 48))
          : Math.max(2, Math.round(50 - confidence * 48));
        finalRiskRef.current = risk;
        setLiveRiskScore(risk);
        onStreamingUpdate({ elapsed, chunkNumber: elapsed + 1, isSynthetic: synth, currentRisk: risk });
      } catch {
        const synth = callSimulationType === 'deepfake';
        const risk = synth
          ? Math.min(96, finalRiskRef.current + 5)
          : Math.max(4, finalRiskRef.current - 2);
        finalRiskRef.current = risk;
        setLiveRiskScore(risk);
        onStreamingUpdate({ elapsed, chunkNumber: elapsed + 1, isSynthetic: synth, currentRisk: risk });
      }
    } catch (err) {
      console.warn('Chunk no decodificable, se omite:', err);
    } finally {
      isSendingRef.current = false;
    }
  };

  const endCall = async () => {
    cleanUpMedia();
    setIsCallActive(false);

    const finalDuration = callDuration;
    const deepfake = callSimulationType === 'deepfake';
    let finalRisk = Math.max(2, Math.min(99, finalRiskRef.current || (deepfake ? 94 : 6)));
    let finalSynth = finalRisk > 50;
    let confidence = finalRisk > 50 ? finalRisk / 100 : (100 - finalRisk) / 100;

    let recordingTitle = `Llamada en Vivo (${finalDuration}s)`;

    // Rebuild the full recorded call and push it into the shared results (waveform + verdict)
    if (micStatus === 'live' && allChunksRef.current.length > 0) {
      try {
        const blob = new Blob(allChunksRef.current, { type: 'audio/webm' });
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioCtx.decodeAudioData(arrayBuffer);
        const resampled = await resampleTo8kHzStereo(decoded);
        const wav = encodeStereoWav8kHz(resampled);
        const b64 = arrayBufferToBase64(wav);
        recordingTitle = `Llamada en Vivo (${resampled.duration.toFixed(1)}s)`;

        if (onRecordingResult) {
          onRecordingResult({
            title: recordingTitle,
            channel0: resampled.getChannelData(0),
            channel1: resampled.getChannelData(1),
            duration: resampled.duration,
            base64: b64
          });
        }

        // Definitive final pass against the streaming endpoint
        try {
          const data = await callDetectStreamingApi(apiBaseUrl, {
            audio: b64,
            chunk_index: 'final',
            sample_rate: 8000
          });
          const conf = Math.min(1, Math.max(0, Number(data?.confidence ?? 0.5)));
          const synth = Boolean(data?.is_synthetic);
          finalRisk = synth ? Math.min(99, Math.round(50 + conf * 48)) : Math.max(2, Math.round(50 - conf * 48));
          finalSynth = synth;
          confidence = conf;
        } catch (err) {
          console.warn('Veredicto final por simulación (API no disponible):', err);
        }
      } catch (err) {
        console.warn('No se pudo reconstruir la grabación del micrófono:', err);
      }
    }

    const summary = {
      duration: finalDuration,
      chunksTotal: streamChunksSent,
      isSynthetic: finalSynth,
      confidence,
      finalRisk,
      title: recordingTitle
    };

    onStreamingUpdate({
      elapsed: finalDuration,
      chunkNumber: streamChunksSent,
      isSynthetic: finalSynth,
      currentRisk: finalRisk
    });

    onCallFinished(summary);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const micPill = () => {
    if (micStatus === 'live') {
      return (
        <span className="altur-badge badge-emerald">
          <span className="dot-pulse green"></span>
          <span>Micrófono en Vivo · Análisis por Chunks</span>
        </span>
      );
    }
    if (micStatus === 'fallback') {
      return (
        <span className="altur-badge badge-rose">
          <MicOff size={12} />
          <span>Sin Micrófono · Simulación</span>
        </span>
      );
    }
    return (
      <span className="altur-badge" style={{ background: '#e2e8f0', color: '#64748b' }}>
        <Mic size={12} />
        <span>Micrófono no solicitado</span>
      </span>
    );
  };

  return (
    <div className="avant-card" style={{ padding: '28px', display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'var(--accent-rose-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Radio size={18} color="#e11d48" />
            </div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--altur-black)', letterSpacing: '-0.02em' }}>
              Live Call · Detección en Streaming
            </h2>
          </div>
          <span className="altur-badge badge-rose font-mono">
            POST /detect_streaming
          </span>
        </div>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          Graba la llamada en tiempo real desde el micrófono y transmite fragmentos (chunks) de 8kHz estéreo
          al endpoint de streaming para evaluar al llamante de forma progresiva.
        </p>
      </div>

      {/* Target Caller Selector (fallback simulation scenario) */}
      {!isCallActive && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-muted)' }}>
            Escenario de respaldo de demostración (se usa si la API o el micrófono no están disponibles):
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <button
              className="btn-altur-outline"
              style={{
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                border: callSimulationType === 'deepfake' ? '2px solid #e11d48' : '1px solid var(--border-card)',
                background: callSimulationType === 'deepfake' ? 'var(--accent-rose-light)' : '#ffffff',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                textAlign: 'left'
              }}
              onClick={() => setCallSimulationType('deepfake')}
            >
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                background: callSimulationType === 'deepfake' ? '#e11d48' : '#f1f5f9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <UserX size={18} color={callSimulationType === 'deepfake' ? '#ffffff' : '#64748b'} />
              </div>
              <div>
                <div style={{ fontWeight: 800, fontSize: '0.86rem', color: 'var(--altur-black)' }}>Ataque Deepfake</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Voz generada por IA</div>
              </div>
            </button>

            <button
              className="btn-altur-outline"
              style={{
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                border: callSimulationType === 'human' ? '2px solid #059669' : '1px solid var(--border-card)',
                background: callSimulationType === 'human' ? 'var(--accent-emerald-light)' : '#ffffff',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                textAlign: 'left'
              }}
              onClick={() => setCallSimulationType('human')}
            >
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                background: callSimulationType === 'human' ? '#059669' : '#f1f5f9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <UserCheck size={18} color={callSimulationType === 'human' ? '#ffffff' : '#64748b'} />
              </div>
              <div>
                <div style={{ fontWeight: 800, fontSize: '0.86rem', color: 'var(--altur-black)' }}>Cliente Legítimo</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Voz humana real</div>
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Live Call Interface */}
      <div style={{
        background: '#f8fafc',
        border: '1px solid var(--border-card)',
        borderRadius: 'var(--radius-md)',
        padding: '22px',
        display: 'flex',
        flexDirection: 'column',
        gap: '18px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Llamada Telefónica Bancaria Altur
            </span>
          </div>
          {isCallActive ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              {micPill()}
              <div className="altur-badge badge-rose">
                <span className="dot-pulse red"></span>
                <span>EN VIVO ({formatTime(callDuration)})</span>
              </div>
            </div>
          ) : (
            <span className="altur-badge" style={{ background: '#e2e8f0', color: '#64748b' }}>
              En Espera
            </span>
          )}
        </div>

        {/* Dual Speakers */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          {/* Agent Box */}
          <div style={{
            background: currentSpeaker === 'agent' && isCallActive ? '#ffffff' : 'rgba(255, 255, 255, 0.6)',
            border: `1px solid ${currentSpeaker === 'agent' && isCallActive ? 'var(--accent-violet)' : 'var(--border-card)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '10px',
            boxShadow: currentSpeaker === 'agent' && isCallActive ? 'var(--shadow-md)' : 'none',
            transition: 'all 0.3s ease'
          }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '50%',
              background: 'var(--accent-violet-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px solid var(--accent-violet)'
            }}>
              <Volume2 size={22} color="#6366f1" />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--altur-black)' }}>
                Agente Altur
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                {isCallActive && currentSpeaker === 'agent' ? 'Hablando ahora...' : 'Canal 1 (8kHz)'}
              </div>
            </div>
            {isCallActive && currentSpeaker === 'agent' && (
              <div className="voice-wave-bars">
                <div className="voice-wave-bar" style={{ background: '#6366f1' }}></div>
                <div className="voice-wave-bar" style={{ background: '#6366f1' }}></div>
                <div className="voice-wave-bar" style={{ background: '#6366f1' }}></div>
                <div className="voice-wave-bar" style={{ background: '#6366f1' }}></div>
                <div className="voice-wave-bar" style={{ background: '#6366f1' }}></div>
              </div>
            )}
          </div>

          {/* Caller Box */}
          <div style={{
            background: currentSpeaker === 'caller' && isCallActive ? '#ffffff' : 'rgba(255, 255, 255, 0.6)',
            border: `1px solid ${currentSpeaker === 'caller' && isCallActive ? 'var(--accent-cyan)' : 'var(--border-card)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '10px',
            boxShadow: currentSpeaker === 'caller' && isCallActive ? 'var(--shadow-md)' : 'none',
            transition: 'all 0.3s ease'
          }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '50%',
              background: micStatus === 'live' ? 'var(--accent-emerald-light)' : 'var(--accent-cyan-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: `2px solid ${micStatus === 'live' ? 'var(--accent-emerald)' : 'var(--accent-cyan)'}`
            }}>
              {micStatus === 'live' ? <Mic size={22} color="#059669" /> : <Mic size={22} color="#0284c7" />}
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--altur-black)' }}>
                Llamante Entrante
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                {isCallActive && currentSpeaker === 'caller' ? 'Hablando ahora...' : 'Canal 0 (Caller)'}
              </div>
            </div>
            {isCallActive && currentSpeaker === 'caller' && (
              <div className="voice-wave-bars">
                <div className="voice-wave-bar" style={{ background: '#0284c7' }}></div>
                <div className="voice-wave-bar" style={{ background: '#0284c7' }}></div>
                <div className="voice-wave-bar" style={{ background: '#0284c7' }}></div>
                <div className="voice-wave-bar" style={{ background: '#0284c7' }}></div>
                <div className="voice-wave-bar" style={{ background: '#0284c7' }}></div>
              </div>
            )}
            {/* Live input level meter */}
            {isCallActive && micStatus === 'live' && (
              <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                <div style={{ height: '4px', background: '#e2e8f0', borderRadius: '999px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.round(micLevel * 100)}%`,
                    background: micLevel > 0.6 ? 'linear-gradient(90deg, #059669, #e11d48)' : 'linear-gradient(90deg, #0284c7, #059669)',
                    transition: 'width 0.08s linear'
                  }} />
                </div>
                <span style={{ fontSize: '0.6rem', color: 'var(--text-faint)', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
                  NIVEL DE ENTRADA · ANALIZADOR
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Live Streaming Risk Bar */}
        {isCallActive && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: '#ffffff', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-card)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
              <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Riesgo de Síntesis en Streaming:</span>
              <span className="font-mono" style={{
                fontWeight: 800,
                color: liveRiskScore > 50 ? '#e11d48' : '#059669'
              }}>
                {liveRiskScore}% {liveRiskScore > 50 ? '(ALERTA DEEPFAKE)' : '(HUMANO VERIFICADO)'}
              </span>
            </div>
            <div style={{
              height: '8px',
              background: '#e2e8f0',
              borderRadius: '999px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                width: `${liveRiskScore}%`,
                background: liveRiskScore > 50
                  ? 'linear-gradient(90deg, #f59e0b 0%, #e11d48 100%)'
                  : 'linear-gradient(90deg, #0284c7 0%, #059669 100%)',
                transition: 'width 0.4s ease'
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              <span>Chunks enviados: #{streamChunksSent} ({callSimulationType === 'deepfake' ? 'Deepfake' : 'Humano'})</span>
              <span className="font-mono">Frecuencia de análisis: ~1 chunk/s</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Buttons */}
      {!isCallActive ? (
        <button
          className="btn-altur btn-altur-primary"
          style={{ width: '100%', padding: '16px', fontSize: '1rem' }}
          onClick={startCall}
        >
          <Phone size={18} />
          <span>Iniciar Llamada en Vivo (Micrófono)</span>
        </button>
      ) : (
        <button
          className="btn-altur btn-altur-rose"
          style={{ width: '100%', padding: '16px', fontSize: '1rem' }}
          onClick={endCall}
        >
          <PhoneOff size={18} />
          <span>Finalizar Llamada y Obtener Veredicto</span>
        </button>
      )}
    </div>
  );
}