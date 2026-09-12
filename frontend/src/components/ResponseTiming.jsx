import React from 'react';
import { Clock, MessageSquareQuote } from 'lucide-react';

// Deterministic (no re-randomization across renders) turn latencies
function buildTurns(isSynthetic) {
  const n = 7;
  if (isSynthetic) {
    return Array.from({ length: n }, (_, i) => 168 + ((i * 37) % 23)); // flat ~180ms
  }
  return Array.from({ length: n }, (_, i) => Math.round(320 + Math.abs(Math.sin(i * 2.1)) * 240 + ((i * 53) % 45))); // organic 320-620ms
}

const X_MAX = 700;

export default function ResponseTiming({ isSynthetic, turnRecoveryMs }) {
  const turns = buildTurns(Boolean(isSynthetic));
  const baseMs = turnRecoveryMs ?? (isSynthetic ? 180 : 420);

  const M = { l: 52, r: 34, t: 16, b: 26 };
  const W = 620;
  const H = 216;
  const plotW = W - M.l - M.r;
  const plotH = H - M.t - M.b;
  const rowH = plotH / turns.length;

  const xOf = (ms) => M.l + (ms / X_MAX) * plotW;

  return (
    <div className="avant-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Clock size={18} color="#d97706" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            Timing de Respuesta
          </h3>
        </div>
        <span className={`altur-badge ${isSynthetic ? 'badge-rose' : 'badge-emerald'}`}>
          {isSynthetic ? 'Patrón plano detectado' : 'Variación natural detectada'}
        </span>
      </div>

      <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        Latencias de recuperación del llamante ante las interrupciones del agente. Un humano tarda entre
        250–600ms (vacilación natural); un pipeline de síntesis responde con latencia casi constante.
      </p>

      {/* Bar Chart */}
      <div style={{
        background: '#f8fafc',
        border: '1px solid var(--border-card)',
        borderRadius: 'var(--radius-sm)',
        padding: '8px',
        overflowX: 'auto'
      }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: '420px', display: 'block' }}>
          {/* Grid + axis labels */}
          {[0, 175, 350, 525, 700].map((ms) => (
            <g key={ms}>
              <line
                x1={xOf(ms)}
                y1={M.t}
                x2={xOf(ms)}
                y2={H - M.b}
                stroke={ms === 0 ? '#cbd5e1' : '#e8edf3'}
                strokeWidth={1}
              />
              <text
                x={xOf(ms)}
                y={H - 8}
                textAnchor="middle"
                fontSize="9"
                fill="#94a3b8"
                fontFamily="JetBrains Mono, monospace"
                fontWeight={700}
              >
                {ms}ms
              </text>
            </g>
          ))}

          {/* Human threshold reference line */}
          <line
            x1={xOf(260)}
            y1={M.t}
            x2={xOf(260)}
            y2={H - M.b}
            stroke="#d97706"
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
          <text
            x={xOf(260) + 4}
            y={M.t + 10}
            fontSize="9"
            fill="#d97706"
            fontFamily="JetBrains Mono, monospace"
            fontWeight={700}
          >
            umbral humano 260ms
          </text>

          {/* Bars */}
          {turns.map((ms, i) => {
            const y = M.t + i * rowH + (rowH / 2 - 10);
            const humanLike = ms >= 260;
            return (
              <g key={i}>
                <text
                  x={M.l - 8}
                  y={y + 11}
                  textAnchor="end"
                  fontSize="9.5"
                  fill="#64748b"
                  fontFamily="Outfit, sans-serif"
                  fontWeight={800}
                >
                  Turno {i + 1}
                </text>
                <rect
                  x={xOf(0)}
                  y={y}
                  width={Math.max(4, xOf(ms) - xOf(0))}
                  height={20}
                  rx={5}
                  fill={humanLike ? 'url(#rtGradEmerald)' : 'url(#rtGradRose)'}
                />
                <text
                  x={xOf(ms) + 6}
                  y={y + 13.5}
                  fontSize="10"
                  fill={humanLike ? '#059669' : '#e11d48'}
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight={800}
                >
                  {ms}ms
                </text>
              </g>
            );
          })}

          <defs>
            <linearGradient id="rtGradEmerald" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
            <linearGradient id="rtGradRose" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#e11d48" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Verdict hint */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        background: isSynthetic ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)',
        border: `1px solid ${isSynthetic ? 'rgba(225, 29, 72, 0.2)' : 'rgba(5, 150, 105, 0.2)'}`,
        borderRadius: 'var(--radius-sm)',
        padding: '12px 14px'
      }}>
        <MessageSquareQuote size={16} color={isSynthetic ? '#e11d48' : '#059669'} />
        <span style={{ fontSize: '0.77rem', color: 'var(--text-primary)', lineHeight: 1.4, fontWeight: 600 }}>
          {isSynthetic
            ? `Recuperación casi invariante (≈${baseMs}ms): firma típica de pipeline STT → LLM → TTS.`
            : `Recuperación orgánica variable (≈${baseMs}ms): el llamante duda, se corrige y respira entre turnos.`}
        </span>
      </div>
    </div>
  );
}