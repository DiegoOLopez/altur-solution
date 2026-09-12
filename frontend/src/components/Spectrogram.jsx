import React, { useMemo, useRef, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { computeSpectrogram } from '../utils/audioUtils';

// Colormap stops tied to the Altur palette (light -> cyan -> violet -> rose)
const STOPS = [
  [0.0, '#f8fafc'],
  [0.2, '#e0f2fe'],
  [0.45, '#7dd3fc'],
  [0.7, '#0284c7'],
  [0.85, '#6366f1'],
  [1.0, '#e11d48']
];

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function colormap(v) {
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [v0, c0] = STOPS[i];
    const [v1, c1] = STOPS[i + 1];
    if (v >= v0 && v <= v1) {
      const t = (v - v0) / (v1 - v0);
      const a = hexToRgb(c0);
      const b = hexToRgb(c1);
      const r = Math.round(a[0] + (b[0] - a[0]) * t);
      const g = Math.round(a[1] + (b[1] - a[1]) * t);
      const bl = Math.round(a[2] + (b[2] - a[2]) * t);
      return `rgb(${r},${g},${bl})`;
    }
  }
  return STOPS[STOPS.length - 1][1];
}


export default function Spectrogram({ channelData, sampleRate = 8000, duration, isAnalyzing }) {
  const canvasRef = useRef(null);
  const W = 800;
  const H = 200;

  const spec = useMemo(() => computeSpectrogram(channelData, sampleRate), [channelData, sampleRate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, W, H);

    if (!spec) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 13px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Espectrograma disponible cuando haya audio del llamante', W / 2, H / 2);
      return;
    }



    // Background grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let f = 0; f <= 4000; f += 1000) {
      const y = H - Math.round((f / spec.maxFreq) * H);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
    for (let t = 0; t <= spec.duration; t += 1) {
      if (t === 0) continue;
      const x = Math.round((t / spec.duration) * W);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    // Spectrogram pixels
    for (let x = 0; x < W; x++) {
      const frame = Math.min(spec.frames - 1, Math.floor((x / W) * spec.frames));
      for (let y = 0; y < H; y++) {
        const col = Math.min(spec.bins - 1, Math.floor((y / H) * spec.bins));
        const row = spec.bins - 1 - col;
        const v = spec.data[frame * spec.bins + row];
        ctx.fillStyle = colormap(v);
        ctx.fillRect(x, y, 1, 1);
      }
    }


    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '700 10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    for (let f = 0; f <= 4000; f += 1000) {
      ctx.fillText(`${f / 1000}kHz`, 6, H - Math.round((f / spec.maxFreq) * H) - 4);
    }
  }, [spec]);

  return (
    <div className="avant-card fx-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <span className="fx-kicker">Audio del llamante · Tiempo-frecuencia</span>
          <h3 className="fx-card-title">Espectrograma del Llamante</h3>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <span className="altur-badge badge-cyan font-mono">
            STFT 32ms
          </span>
          <span className="altur-badge" style={{ background: '#e2e8f0', color: '#64748b' }}>
            Canal 0 · 0–4 kHz
          </span>
        </div>
      </div>

      <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        Representación tiempo-frecuencia del audio del llamante mediante una transformada de Fourier de tiempo corto.
      </p>
      {/* Spectrogram Canvas */}
      <div style={{
        position: 'relative',
        width: '100%',
        borderRadius: 'var(--radius-sm)',
        overflow: 'hidden',
        border: '1px solid var(--border-card)'
      }}>
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          style={{ width: '100%', height: `${H}px`, display: 'block', background: '#f8fafc' }}
        />
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
          border: '1px solid rgba(2, 132, 199, 0.2)',
          display: 'flex',
          alignItems: 'center',
          gap: '5px'
        }}>
          <Activity size={12} />
          {isAnalyzing ? 'Analizando banda ancha de voz...' : spec ? 'Eje X: tiempo • Eje Y: frecuencia' : 'Sin audio'}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Ventana Hann · {spec ? `${Math.round((spec.winSize / spec.sampleRate) * 1000)}ms` : '—'} · hop {spec ? spec.hop : '—'} muestras
        </span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {duration ? `${duration.toFixed(2)}s` : '—'} de audio del llamante
        </span>
      </div>
    </div>
  );
}