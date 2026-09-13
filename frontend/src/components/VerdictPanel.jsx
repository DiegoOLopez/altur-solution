/**
 * AuraVoice - Panel de Veredicto (VerdictPanel.jsx)
 * 
 * Muestra el resultado de la detección (tanto para análisis batch como
 * streaming). Visualiza el veredicto (Humano / Sintético), la confianza
 * del modelo, el desglose de evidencias (LLR acústico y comportamental)
 * y una conclusión semántica explicativa generada para el usuario.
 */
import React from 'react';
import { ShieldCheck, ShieldAlert, ShieldQuestion, AlertOctagon, Waves, Clock, MessageSquareQuote, Sparkles } from 'lucide-react';
import ChameleonLogo from './ChameleonLogo';

export default function VerdictPanel({ result, isAnalyzing, activeMode }) {
  const isError = Boolean(result && result.error);

  if (isAnalyzing) {
    return (
      <div className="avant-card fx-card" style={{
        padding: '44px 24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '18px',
        textAlign: 'center'
      }}>
        <div className="radar-sweep-container" style={{ color: '#0284c7' }}>
          <div className="radar-ring"></div>
          <div className="radar-ring"></div>
          <div className="radar-ring"></div>
          <ChameleonLogo size={44} showWordmark={false} />
        </div>
        <div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--altur-black)' }}>
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

  if (isError) {
    return (
      <div className="avant-card fx-card" style={{
        padding: '28px',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        alignItems: 'center',
        textAlign: 'center',
        border: '1px solid rgba(225, 29, 72, 0.35)'
      }}>
        <div style={{
          width: '58px',
          height: '58px',
          borderRadius: '50%',
          background: 'var(--accent-rose-light)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <AlertOctagon size={26} color="#e11d48" />
        </div>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--altur-black)' }}>
          Error en el Análisis
        </h3>
        <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', lineHeight: 1.5, maxWidth: '420px' }}>
          {result?.errorMessage ||
            'No se pudo completar el análisis. Intenta nuevamente con otro archivo.'}
        </p>
        <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.5, maxWidth: '420px' }}>
          Verifica que el backend esté activo y que el audio sea un WAV válido.
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="avant-card fx-card" style={{
        padding: '40px 24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '16px',
        textAlign: 'center',
        borderStyle: 'dashed'
      }}>
        <div style={{
          width: '58px',
          height: '58px',
          borderRadius: '50%',
          background: '#f8fafc',
          border: '1px solid var(--border-card)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <ShieldQuestion size={26} color="#64748b" />
        </div>
        <div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            Veredicto Pendiente
          </h3>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.5, maxWidth: '420px' }}>
            Selecciona o sube un audio del llamante y ejecuta POST /detect
            para obtener la clasificación del modelo.
          </p>
        </div>
      </div>
    );
  }

  const isSynthetic = result.is_synthetic;
  const confidencePct =
    Math.round((result.confidence || 0) * 1000) / 10;

  return (
    <div className="avant-card fx-card" style={{
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '18px',
      border: isSynthetic ? '1px solid rgba(225, 29, 72, 0.4)' : '1px solid rgba(5, 150, 105, 0.4)',
      boxShadow: isSynthetic ? 'var(--shadow-glow-rose)' : 'var(--shadow-glow-emerald)'
    }}>
      {/* Kicker + verdict badge */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '10px'
      }}>
        <div>
          <span className="fx-kicker">Veredicto del modelo</span>
          <div
            style={{
              display: 'inline-block',
              fontSize: '0.72rem',
              fontWeight: 800,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: isSynthetic ? '#e11d48' : '#059669',
              marginTop: '4px'
            }}
          >
            {isSynthetic ? 'Posible Voz Sintética' : 'Voz Clasificada como Humana'}
          </div>
        </div>
        <span className={`altur-badge ${isSynthetic ? 'badge-rose' : 'badge-emerald'}`}>
          {isSynthetic ? 'POST /detect → sintética' : 'POST /detect → humana'}
        </span>
      </div>

      {/* Hero Verdict Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        padding: '18px',
        borderRadius: 'var(--radius-md)',
        background: isSynthetic ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)',
        border: `1px solid ${isSynthetic ? 'rgba(225, 29, 72, 0.2)' : 'rgba(5, 150, 105, 0.2)'}`
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div className="radar-sweep-container" style={{
            color: isSynthetic ? '#e11d48' : '#059669',
            width: '60px',
            height: '60px'
          }}>
            <div className="radar-ring"></div>
            <div className="radar-ring"></div>
            <div style={{
              width: '50px',
              height: '50px',
              borderRadius: '50%',
              background: isSynthetic ? '#e11d48' : '#059669',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: isSynthetic ? 'var(--shadow-glow-rose)' : 'var(--shadow-glow-emerald)'
            }}>
              {isSynthetic ? <ShieldAlert size={26} color="#ffffff" /> : <ShieldCheck size={26} color="#ffffff" />}
            </div>
          </div>

          <div>
            <h2 style={{
              fontSize: '1.5rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              color: 'var(--altur-black)',
              lineHeight: 1.2
            }}>
              {isSynthetic ? 'Voz Sintética' : 'Voz Humana'}
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              {isSynthetic
                ? 'El modelo clasificó la voz como sintética.'
                : 'El modelo clasificó la voz como humana.'}
            </p>
          </div>
        </div>

        {/* Confidence Percentage Tag */}
        <div style={{
          textAlign: 'right',
          background: '#ffffff',
          padding: '12px 20px',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-card)',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <div className="font-mono" style={{
            fontSize: '2rem',
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

      {/* Real model evidence */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '14px'
        }}
      >
        {/* Acoustic evidence */}
        <div
          style={{
            background: '#f8fafc',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '16px'
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px'
            }}
          >
            <Waves size={16} color="#0284c7" />

            <span
              style={{
                fontSize: '0.84rem',
                fontWeight: 800
              }}
            >
              Evidencia Acústica
            </span>
          </div>

          <div
            className="font-mono"
            style={{
              fontSize: '1.05rem',
              fontWeight: 800
            }}
          >
            {result.llr_acoustic_cum?.toFixed(2) ?? '—'}
          </div>

          <p
            style={{
              fontSize: '0.74rem',
              color: 'var(--text-secondary)',
              marginTop: '5px'
            }}
          >
            LLR acústico acumulado
          </p>
        </div>


        {/* Behavioral evidence */}
        <div
          style={{
            background: '#f8fafc',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '16px'
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px'
            }}
          >
            <Clock size={16} color="#6366f1" />

            <span
              style={{
                fontSize: '0.84rem',
                fontWeight: 800
              }}
            >
              Evidencia Conversacional
            </span>
          </div>

          <div
            className="font-mono"
            style={{
              fontSize: '1.05rem',
              fontWeight: 800
            }}
          >
            {result.llr_behavioral_cum?.toFixed(2) ?? '—'}
          </div>

          <p
            style={{
              fontSize: '0.74rem',
              color: 'var(--text-secondary)',
              marginTop: '5px'
            }}
          >
            LLR comportamental acumulado
          </p>
        </div>


        {/* Analysis volume */}
        <div
          style={{
            background: '#f8fafc',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '16px'
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px'
            }}
          >
            <Sparkles size={16} color="#d97706" />

            <span
              style={{
                fontSize: '0.84rem',
                fontWeight: 800
              }}
            >
              Profundidad del Análisis
            </span>
          </div>

          <div
            className="font-mono"
            style={{
              fontSize: '1.05rem',
              fontWeight: 800
            }}
          >
            {result.n_acoustic_segments ?? 0}
            <span
              style={{
                fontSize: '0.75rem',
                fontWeight: 600
              }}
            >
              {' '}segmentos
            </span>
          </div>

          <p
            style={{
              fontSize: '0.74rem',
              color: 'var(--text-secondary)',
              marginTop: '5px'
            }}
          >
            {result.n_behavioral_events ?? 0} eventos conversacionales analizados
          </p>
        </div>
      </div>

      {/* Real inference metadata chips */}
      {(result.eta !== undefined || result.latency_ms !== undefined || result.apiResponse) && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {result.eta !== undefined && (
            <span className="altur-badge font-mono" style={{ background: '#f1f5f9', color: '#334155' }}>
              Umbral del detector (η): {typeof result.eta === 'number' ? result.eta.toFixed(2) : result.eta}
            </span>
          )}
          {result.latency_ms !== undefined && (
            <span className="altur-badge font-mono" style={{ background: '#f1f5f9', color: '#334155' }}>
              Latencia: {result.latency_ms} ms
            </span>
          )}
          {result.apiResponse?.message && (
            <span className="altur-badge" style={{ background: '#f1f5f9', color: '#334155' }}>
              {result.apiResponse.message}
            </span>
          )}
        </div>
      )}

      {/* Friendly LLM Explanation */}
      <div style={{
        background: '#f8fafc',
        border: '1px solid #cbd5e1',
        borderRadius: 'var(--radius-sm)',
        padding: '18px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MessageSquareQuote size={18} color="#0284c7" />
          <h4 style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--altur-black)' }}>
            Interpretación del Modelo:
          </h4>
        </div>
        <p style={{
          fontSize: '0.88rem',
          lineHeight: 1.6,
          color: 'var(--text-primary)',
          whiteSpace: 'pre-line'
        }}>
          {result.llm_conclusion || 'Sin interpretación disponible para este análisis.'}
        </p>
      </div>
    </div>
  );
}