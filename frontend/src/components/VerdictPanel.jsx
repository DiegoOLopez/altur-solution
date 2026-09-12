import React from 'react';
import { ShieldCheck, ShieldAlert, Waves, Clock, MessageSquareQuote, Sparkles, Check, X, Bot, AlertOctagon } from 'lucide-react';
import ChameleonLogo from './ChameleonLogo';

export default function VerdictPanel({ result, isAnalyzing, activeMode }) {
  if (isAnalyzing) {
    return (
      <div className="avant-card" style={{
        padding: '50px 24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '20px',
        textAlign: 'center'
      }}>
        <div className="radar-sweep-container" style={{ color: '#0284c7' }}>
          <div className="radar-ring"></div>
          <div className="radar-ring"></div>
          <div className="radar-ring"></div>
          <ChameleonLogo size={44} showWordmark={false} />
        </div>
        <div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            {activeMode === 'streaming'
              ? 'Procesando streaming en POST /detect_streaming...'
              : 'Ejecutando inferencia en POST /detect...'}
          </h3>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Extrayendo artefactos espectrales, latencia de interrupción y consistencia semántica...
          </p>
        </div>
      </div>
    );
  }

  const isSynthetic = result ? result.is_synthetic : false;
  const confidencePct = Math.round((result?.confidence || 0.96) * 1000) / 10;
  const metrics = result?.metrics || {
    acoustic_score: isSynthetic ? 93 : 8,
    spectral_artifacts: isSynthetic ? 'Artefactos vocoder neural en 3.8 kHz.' : 'Voz humana con atenuación natural.',
    turn_recovery_ms: isSynthetic ? 180 : 420,
    conversational_messiness: isSynthetic ? 12 : 88,
    breathing_detected: !isSynthetic,
    semantic_hallucination: isSynthetic
  };

  return (
    <div className="avant-card" style={{
      padding: '28px',
      display: 'flex',
      flexDirection: 'column',
      gap: '22px',
      border: isSynthetic ? '2px solid rgba(225, 29, 72, 0.3)' : '2px solid rgba(5, 150, 105, 0.3)',
      boxShadow: isSynthetic ? 'var(--shadow-glow-rose)' : 'var(--shadow-glow-emerald)'
    }}>
      {/* Hero Verdict Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '20px',
        padding: '22px',
        borderRadius: 'var(--radius-md)',
        background: isSynthetic ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)',
        border: `1px solid ${isSynthetic ? 'rgba(225, 29, 72, 0.2)' : 'rgba(5, 150, 105, 0.2)'}`
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="radar-sweep-container" style={{
            color: isSynthetic ? '#e11d48' : '#059669',
            width: '64px',
            height: '64px'
          }}>
            <div className="radar-ring"></div>
            <div className="radar-ring"></div>
            <div style={{
              width: '54px',
              height: '54px',
              borderRadius: '50%',
              background: isSynthetic ? '#e11d48' : '#059669',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: isSynthetic ? 'var(--shadow-glow-rose)' : 'var(--shadow-glow-emerald)'
            }}>
              {isSynthetic ? <ShieldAlert size={28} color="#ffffff" /> : <ShieldCheck size={28} color="#ffffff" />}
            </div>
          </div>

          <div>
            <div style={{
              display: 'inline-block',
              fontSize: '0.72rem',
              fontWeight: 800,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: isSynthetic ? '#e11d48' : '#059669',
              marginBottom: '2px'
            }}>
              {isSynthetic ? 'Alerta de Suplantación Telefónica' : 'Identidad Biométrica Confirmada'}
            </div>
            <h2 style={{
              fontSize: '1.5rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              color: 'var(--altur-black)',
              lineHeight: 1.2
            }}>
              {isSynthetic ? 'Voz Sintética (Deepfake)' : 'Voz Humana Auténtica'}
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              {isSynthetic
                ? 'El llamante utiliza un sintetizador o clonador de voz neuronal con IA.'
                : 'El llamante presenta modulación biológica natural sin indicios de manipulación.'}
            </p>
          </div>
        </div>

        {/* Confidence Percentage Tag */}
        <div style={{
          textAlign: 'right',
          background: '#ffffff',
          padding: '14px 22px',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-card)',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <div className="font-mono" style={{
            fontSize: '2.2rem',
            fontWeight: 800,
            color: isSynthetic ? '#e11d48' : '#059669',
            lineHeight: 1
          }}>
            {confidencePct}%
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px', textTransform: 'uppercase', fontWeight: 700 }}>
            Nivel de Confianza
          </div>
        </div>
      </div>

      {/* 3 Clear Signal Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
        {/* Signal 1: Acoustic */}
        <div style={{
          background: '#f8fafc',
          border: '1px solid var(--border-card)',
          borderRadius: 'var(--radius-sm)',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.84rem', fontWeight: 800, color: '#0284c7', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Waves size={16} />
              1. Tono y Acústica
            </span>
            <span className="altur-badge" style={{
              background: metrics.breathing_detected ? 'var(--accent-emerald-light)' : 'var(--accent-rose-light)',
              color: metrics.breathing_detected ? '#059669' : '#e11d48'
            }}>
              {metrics.breathing_detected ? 'Respiración OK' : 'Robótico'}
            </span>
          </div>
          <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            {metrics.spectral_artifacts}
          </p>
        </div>

        {/* Signal 2: Conversational rhythm */}
        <div style={{
          background: '#f8fafc',
          border: '1px solid var(--border-card)',
          borderRadius: 'var(--radius-sm)',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.84rem', fontWeight: 800, color: '#6366f1', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Clock size={16} />
              2. Ritmo de Diálogo
            </span>
            <span className="font-mono altur-badge badge-violet">
              {metrics.turn_recovery_ms}ms
            </span>
          </div>
          <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            {isSynthetic
              ? 'Pausa fija e invariable tras interrupciones del agente bancario.'
              : 'Vacilación y tiempo de reacción natural de una persona real.'}
          </p>
        </div>

        {/* Signal 3: Semantic coherence */}
        <div style={{
          background: '#f8fafc',
          border: '1px solid var(--border-card)',
          borderRadius: 'var(--radius-sm)',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.84rem', fontWeight: 800, color: '#d97706', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sparkles size={16} />
              3. Coherencia Semántica
            </span>
            <span className="altur-badge" style={{
              background: metrics.semantic_hallucination ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)',
              color: metrics.semantic_hallucination ? '#e11d48' : '#059669'
            }}>
              {metrics.semantic_hallucination ? 'Alucinación' : 'Coherente'}
            </span>
          </div>
          <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            {metrics.semantic_hallucination
              ? 'El modelo inventó datos ante una pregunta trampa inexistente.'
              : 'El interlocutor respondió con naturalidad.'}
          </p>
        </div>
      </div>

      {/* Friendly LLM Explanation */}
      <div style={{
        background: '#f8fafc',
        border: '1px solid #cbd5e1',
        borderRadius: 'var(--radius-sm)',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MessageSquareQuote size={18} color="#0284c7" />
          <h4 style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            Dictamen Forense Explicable (LLM):
          </h4>
        </div>
        <p style={{
          fontSize: '0.88rem',
          lineHeight: 1.6,
          color: 'var(--text-primary)',
          whiteSpace: 'pre-line'
        }}>
          {result?.llm_conclusion || 'Listo para procesar la llamada telefónica.'}
        </p>
      </div>
    </div>
  );
}
