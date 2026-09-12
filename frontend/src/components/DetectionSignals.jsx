import React from 'react';
import { Activity, Waves, Clock, Sparkles, HeartPulse, Check, X } from 'lucide-react';

function MeterRow({ label, value, display, fillColor, barColor, chip, chipColor, bg }) {
  const pct = Math.min(100, Math.max(0, Math.round(value)));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--altur-black)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          {label}
        </span>
        {chip && (
          <span className="altur-badge font-mono" style={{ background: bg, color: chipColor, fontSize: '0.62rem', padding: '2px 9px' }}>
            {chip}
          </span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div style={{ flex: 1, height: '7px', background: '#e2e8f0', borderRadius: '999px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${pct}%`,
            background: fillColor,
            transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
          }} />
        </div>
        <span className="font-mono" style={{ fontSize: '0.78rem', fontWeight: 800, color: barColor, minWidth: '46px', textAlign: 'right' }}>
          {display}
        </span>
      </div>
    </div>
  );
}

export default function DetectionSignals({ metrics, isSynthetic }) {
  const m = metrics || {
    acoustic_score: isSynthetic ? 93 : 8,
    spectral_artifacts: isSynthetic ? 'Artefactos vocoder neural en 3.8 kHz.' : 'Voz humana con atenuación natural.',
    turn_recovery_ms: isSynthetic ? 180 : 420,
    conversational_messiness: isSynthetic ? 12 : 88,
    breathing_detected: !isSynthetic,
    semantic_hallucination: isSynthetic
  };

  const ac = Math.min(100, m.acoustic_score ?? 50);
  const turnPct = Math.min(100, ((m.turn_recovery_ms ?? 420) / 600) * 100);
  const messiness = Math.min(100, m.conversational_messiness ?? 50);
  const breathing = Boolean(m.breathing_detected);
  const hallucination = Boolean(m.semantic_hallucination);

  const humanTurn = (m.turn_recovery_ms ?? 420) >= 260;

  return (
    <div className="avant-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} color="#6366f1" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            Señales de Detección
          </h3>
        </div>
        <span className={`altur-badge ${isSynthetic ? 'badge-rose' : 'badge-emerald'}`}>
          {isSynthetic ? '3/3 señales anómalas' : '3/3 señales naturales'}
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', background: '#f8fafc', border: '1px solid var(--border-card)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
        {/* Signal 1: Spectral anomaly */}
        <MeterRow
          label={<><Waves size={14} color="#0284c7" /> 1. Anomalía Espectral</>}
          value={ac}
          display={`${ac}%`}
          fillColor={ac > 50 ? 'linear-gradient(90deg, #f59e0b, #e11d48)' : 'linear-gradient(90deg, #0284c7, #059669)'}
          barColor={ac > 50 ? '#e11d48' : '#059669'}
          chip={ac > 50 ? 'Artefacto presente' : 'Formantes naturales'}
          chipColor={ac > 50 ? '#e11d48' : '#059669'}
          bg={ac > 50 ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)'}
        />

        {/* Signal 2: Turn recovery */}
        <MeterRow
          label={<><Clock size={14} color="#6366f1" /> 2. Recuperación de Turno</>}
          value={turnPct}
          display={`${m.turn_recovery_ms ?? 420}ms`}
          fillColor={humanTurn ? 'linear-gradient(90deg, #6366f1, #059669)' : 'linear-gradient(90deg, #f59e0b, #e11d48)'}
          barColor={humanTurn ? '#059669' : '#e11d48'}
          chip={humanTurn ? 'Vacilación natural' : 'Latencia plana'}
          chipColor={humanTurn ? '#059669' : '#e11d48'}
          bg={humanTurn ? 'var(--accent-emerald-light)' : 'var(--accent-rose-light)'}
        />

        {/* Signal 3: Conversational messiness */}
        <MeterRow
          label={<><Sparkles size={14} color="#d97706" /> 3. Desorden Conversacional</>}
          value={messiness}
          display={`${messiness}/100`}
          fillColor={messiness > 40 ? 'linear-gradient(90deg, #d97706, #059669)' : 'linear-gradient(90deg, #f59e0b, #e11d48)'}
          barColor={messiness > 40 ? '#059669' : '#e11d48'}
          chip={messiness > 40 ? 'Prosodia orgánica' : 'Ritmo robótico'}
          chipColor={messiness > 40 ? '#059669' : '#e11d48'}
          bg={messiness > 40 ? 'var(--accent-emerald-light)' : 'var(--accent-rose-light)'}
        />

        {/* Boolean signals */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          {/* Breathing */}
          <div style={{
            background: '#ffffff',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.76rem', fontWeight: 700, color: 'var(--altur-black)' }}>
              <HeartPulse size={14} color={breathing ? '#059669' : '#e11d48'} />
              Micro-respiración
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.74rem', color: breathing ? '#059669' : '#e11d48', fontWeight: 700 }}>
              {breathing ? <Check size={14} /> : <X size={14} />}
              {breathing ? 'Detectada (biológica)' : 'Ausente (sintética)'}
            </div>
          </div>

          {/* Semantics */}
          <div style={{
            background: '#ffffff',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.76rem', fontWeight: 700, color: 'var(--altur-black)' }}>
              <Sparkles size={14} color={hallucination ? '#e11d48' : '#059669'} />
              Coherencia Semántica
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.74rem', color: hallucination ? '#e11d48' : '#059669', fontWeight: 700 }}>
              {hallucination ? <X size={14} /> : <Check size={14} />}
              {hallucination ? 'Alucinación LLM' : 'Respuestas coherentes'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}