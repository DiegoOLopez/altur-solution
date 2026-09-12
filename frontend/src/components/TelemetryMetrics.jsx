import React from 'react';
import { Waves, Clock, BrainCircuit, Check, X, ShieldAlert } from 'lucide-react';

export default function TelemetryMetrics({ metrics, isSynthetic, isAnalyzing }) {
  const m = metrics || {
    acoustic_score: 93,
    spectral_artifacts: 'Presencia de ruido de fase a 3.8 kHz y armónicos artificialmente estables.',
    turn_recovery_ms: 180,
    conversational_messiness: 12,
    breathing_detected: false,
    semantic_hallucination: true
  };

  return (
    <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Title */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BrainCircuit size={18} color="#00f2fe" />
          <h2 style={{ fontSize: '1rem', fontWeight: '700' }}>
            3. Telemetría de Motores de Análisis (Fase 1)
          </h2>
        </div>
        <span className="chip chip-cyan" style={{ fontSize: '0.65rem' }}>
          3 Señales Altur
        </span>
      </div>

      {/* Grid for the 3 challenge pillars */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
        {/* Pillar 1: Acoustic */}
        <div style={{
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
          padding: '14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#00f2fe', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Waves size={16} />
              1. Detección Acústica
            </span>
            <span className={`chip ${isSynthetic ? 'chip-rose' : 'chip-emerald'}`} style={{ fontSize: '0.62rem' }}>
              {isSynthetic ? 'Artefacto Detectado' : 'Formantes Naturales'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
              <span>Índice de Anomalía Espectral:</span>
              <span className="font-mono" style={{ color: isSynthetic ? '#ef4444' : '#10b981', fontWeight: 700 }}>
                {m.acoustic_score}%
              </span>
            </div>
            <div style={{ height: '6px', width: '100%', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${m.acoustic_score}%`,
                background: isSynthetic ? 'linear-gradient(90deg, #f59e0b, #ef4444)' : 'linear-gradient(90deg, #06b6d4, #10b981)',
                transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
              }} />
            </div>
          </div>

          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            {m.spectral_artifacts}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.72rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '6px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Micro-respiración biológica:</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: m.breathing_detected ? '#10b981' : '#ef4444', fontWeight: 600 }}>
              {m.breathing_detected ? <Check size={12} /> : <X size={12} />}
              {m.breathing_detected ? 'Detectada' : 'Ausente (Sintética)'}
            </span>
          </div>
        </div>

        {/* Pillar 2: Conversational Behavior */}
        <div style={{
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
          padding: '14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#a855f7', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Clock size={16} />
              2. Comportamiento
            </span>
            <span className={`chip ${isSynthetic ? 'chip-rose' : 'chip-emerald'}`} style={{ fontSize: '0.62rem' }}>
              {isSynthetic ? 'Latencia Invariable' : 'Desfase Humano'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
              <span>Recuperación ante Interrupción:</span>
              <span className="font-mono" style={{ color: isSynthetic ? '#ef4444' : '#10b981', fontWeight: 700 }}>
                {m.turn_recovery_ms} ms
              </span>
            </div>
            <div style={{ height: '6px', width: '100%', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${Math.min(100, (m.turn_recovery_ms / 500) * 100)}%`,
                background: isSynthetic ? '#ef4444' : '#a855f7',
                transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
              }} />
            </div>
          </div>

          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            {isSynthetic 
              ? 'Consistencia robótica: El llamante se recupera en un intervalo plano y predecible (típico de pipeline STT+LLM).'
              : 'Fluidez desordenada: Vacilación natural e imperfecciones al retomar la palabra.'}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.72rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '6px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Índice de desorden vocal:</span>
            <span className="font-mono" style={{ color: '#00f2fe', fontWeight: 600 }}>
              {m.conversational_messiness}/100
            </span>
          </div>
        </div>

        {/* Pillar 3: Semantic Integrity */}
        <div style={{
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
          padding: '14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ShieldAlert size={16} />
              3. Verificación Semántica
            </span>
            <span className={`chip ${m.semantic_hallucination ? 'chip-rose' : 'chip-emerald'}`} style={{ fontSize: '0.62rem' }}>
              {m.semantic_hallucination ? 'Alucinación' : 'Coherente'}
            </span>
          </div>

          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            {m.semantic_hallucination
              ? 'El interlocutor inventó una respuesta ante una pregunta trampa inexistente (comportamiento clásico de un LLM).'
              : 'El interlocutor respondió con naturalidad indicando no contar con la información solicitada.'}
          </p>

          <div style={{
            background: 'rgba(0,0,0,0.2)',
            padding: '8px',
            borderRadius: '4px',
            fontSize: '0.68rem',
            color: 'var(--text-muted)'
          }}>
            <span style={{ color: '#f59e0b', fontWeight: 600 }}>Regla Altur:</span> "Un humano dice 'No tengo eso'. Un modelo de lenguaje tiende a inventar."
          </div>
        </div>
      </div>
    </div>
  );
}
