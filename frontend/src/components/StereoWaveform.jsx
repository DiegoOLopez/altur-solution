import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCcw, Activity, AlertCircle, Headphones } from 'lucide-react';

export default function StereoWaveform({ channel0, channel1, duration, isSynthetic, isAnalyzing }) {
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const startTimeRef = useRef(0);
  const animationFrameRef = useRef(null);

  const totalDuration = duration || 6.0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Crisp light background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);

    // Subtle grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal divider
    ctx.strokeStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Channel 0 (Caller) - Top half
    const ch0Center = height / 4;
    const ch0Max = height / 4 - 12;

    if (channel0 && channel0.length > 0) {
      const step = Math.ceil(channel0.length / width);

      // Area fill gradient
      const grad0 = ctx.createLinearGradient(0, 0, 0, height / 2);
      grad0.addColorStop(0, 'rgba(2, 132, 199, 0.22)');
      grad0.addColorStop(1, 'rgba(2, 132, 199, 0.02)');

      ctx.fillStyle = grad0;
      ctx.beginPath();
      ctx.moveTo(0, ch0Center);
      for (let i = 0; i < width; i++) {
        const val = channel0[i * step] || 0;
        ctx.lineTo(i, ch0Center - val * ch0Max);
      }
      ctx.lineTo(width, ch0Center);
      ctx.closePath();
      ctx.fill();

      // Line stroke
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < width; i++) {
        const val = channel0[i * step] || 0;
        const y = ch0Center - val * ch0Max;
        if (i === 0) ctx.moveTo(i, y);
        else ctx.lineTo(i, y);
      }
      ctx.stroke();
    } else {
      ctx.strokeStyle = '#94a3b8';
      ctx.beginPath();
      ctx.moveTo(0, ch0Center);
      ctx.lineTo(width, ch0Center);
      ctx.stroke();
    }

    // Channel 1 (Agent) - Bottom half
    const ch1Center = (height * 3) / 4;
    const ch1Max = height / 4 - 12;

    if (channel1 && channel1.length > 0) {
      const step = Math.ceil(channel1.length / width);

      // Area fill gradient
      const grad1 = ctx.createLinearGradient(0, height / 2, 0, height);
      grad1.addColorStop(0, 'rgba(99, 102, 241, 0.02)');
      grad1.addColorStop(1, 'rgba(99, 102, 241, 0.22)');

      ctx.fillStyle = grad1;
      ctx.beginPath();
      ctx.moveTo(0, ch1Center);
      for (let i = 0; i < width; i++) {
        const val = channel1[i * step] || 0;
        ctx.lineTo(i, ch1Center - val * ch1Max);
      }
      ctx.lineTo(width, ch1Center);
      ctx.closePath();
      ctx.fill();

      // Line stroke
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < width; i++) {
        const val = channel1[i * step] || 0;
        const y = ch1Center - val * ch1Max;
        if (i === 0) ctx.moveTo(i, y);
        else ctx.lineTo(i, y);
      }
      ctx.stroke();
    } else {
      ctx.strokeStyle = '#94a3b8';
      ctx.beginPath();
      ctx.moveTo(0, ch1Center);
      ctx.lineTo(width, ch1Center);
      ctx.stroke();
    }

    // Anomaly highlighter in light theme
    if (isSynthetic && channel0 && channel0.length > 0) {
      const anomalyX = width * 0.52;
      const anomalyWidth = width * 0.22;
      ctx.fillStyle = 'rgba(225, 29, 72, 0.08)';
      ctx.fillRect(anomalyX, 4, anomalyWidth, height / 2 - 8);

      ctx.strokeStyle = '#e11d48';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(anomalyX, 4, anomalyWidth, height / 2 - 8);
      ctx.setLineDash([]);
    }

    // Playhead line
    const playheadX = (currentTime / totalDuration) * width;
    ctx.strokeStyle = '#090d16';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();

    // Playhead indicator dot
    ctx.fillStyle = '#0284c7';
    ctx.beginPath();
    ctx.arc(playheadX, 8, 5, 0, Math.PI * 2);
    ctx.fill();

  }, [channel0, channel1, currentTime, isSynthetic, totalDuration]);

  const togglePlay = () => {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  };

  const startPlayback = () => {
    if (!channel0 || channel0.length === 0) return;

    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const buffer = audioCtx.createBuffer(2, channel0.length, 8000);
      buffer.copyToChannel(channel0, 0);
      if (channel1) buffer.copyToChannel(channel1, 1);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);

      source.onended = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        cancelAnimationFrame(animationFrameRef.current);
      };

      const startOffset = currentTime >= totalDuration ? 0 : currentTime;
      source.start(0, startOffset);
      sourceNodeRef.current = source;
      startTimeRef.current = audioCtx.currentTime - startOffset;
      setIsPlaying(true);

      const updatePlayhead = () => {
        const elapsed = audioCtx.currentTime - startTimeRef.current;
        if (elapsed <= totalDuration) {
          setCurrentTime(elapsed);
          animationFrameRef.current = requestAnimationFrame(updatePlayhead);
        } else {
          setCurrentTime(0);
          setIsPlaying(false);
        }
      };
      animationFrameRef.current = requestAnimationFrame(updatePlayhead);

    } catch (e) {
      console.error('Playback error:', e);
    }
  };

  const stopPlayback = () => {
    if (sourceNodeRef.current) {
      try { sourceNodeRef.current.stop(); } catch (e) {}
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) {}
    }
    setIsPlaying(false);
    cancelAnimationFrame(animationFrameRef.current);
  };

  const handleReset = () => {
    stopPlayback();
    setCurrentTime(0);
  };

  return (
    <div className="avant-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} color="#0284c7" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            Osciloscopio Estéreo de Telefonía
          </h3>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span className="altur-badge badge-cyan">
            Canal 0: Llamante
          </span>
          <span className="altur-badge badge-violet">
            Canal 1: Agente Altur
          </span>
        </div>
      </div>

      {/* Waveform Canvas */}
      <div style={{
        position: 'relative',
        width: '100%',
        borderRadius: 'var(--radius-sm)',
        overflow: 'hidden',
        border: '1px solid var(--border-card)'
      }}>
        <canvas
          ref={canvasRef}
          width={800}
          height={180}
          style={{ width: '100%', height: '180px', display: 'block', background: '#f8fafc' }}
        />

        {/* Labels Overlay */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '12px',
          fontSize: '0.7rem',
          color: '#0284c7',
          fontWeight: 800,
          background: 'rgba(255, 255, 255, 0.9)',
          padding: '3px 8px',
          borderRadius: '6px',
          border: '1px solid rgba(2, 132, 199, 0.2)'
        }}>
          Canal 0 • Voz del Llamante (A Clasificar)
        </div>

        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '12px',
          fontSize: '0.7rem',
          color: '#6366f1',
          fontWeight: 800,
          background: 'rgba(255, 255, 255, 0.9)',
          padding: '3px 8px',
          borderRadius: '6px',
          border: '1px solid rgba(99, 102, 241, 0.2)'
        }}>
          Canal 1 • Agente Altur (Turnos / Interrupciones)
        </div>

        {/* Anomaly Callout Badge */}
        {isSynthetic && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '12px',
            fontSize: '0.7rem',
            color: '#e11d48',
            fontWeight: 800,
            background: 'rgba(255, 241, 242, 0.95)',
            padding: '3px 10px',
            borderRadius: '6px',
            border: '1px solid rgba(225, 29, 72, 0.3)',
            display: 'flex',
            alignItems: 'center',
            gap: '5px'
          }}>
            <AlertCircle size={12} />
            Artefacto de Fase (3.8 kHz)
          </div>
        )}
      </div>

      {/* Playback Controls & Scrubber */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        padding: '10px 14px',
        background: '#f8fafc',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-card)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            className="btn-altur-outline"
            style={{ padding: '8px 16px', fontSize: '0.82rem', gap: '6px' }}
            onClick={togglePlay}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} fill="#0f172a" />}
            <span>{isPlaying ? 'Pausar' : 'Reproducir Audio'}</span>
          </button>
          <button
            className="btn-altur-outline"
            style={{ padding: '8px 10px' }}
            onClick={handleReset}
            title="Reiniciar Posición"
          >
            <RotateCcw size={14} />
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div className="font-mono" style={{ fontSize: '0.86rem', color: '#090d16', fontWeight: 700 }}>
            {currentTime.toFixed(2)}s / {totalDuration.toFixed(2)}s
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Estéreo 8kHz PCM 16-bit
          </span>
        </div>
      </div>
    </div>
  );
}
