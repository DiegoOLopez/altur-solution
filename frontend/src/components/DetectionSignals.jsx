/**
 * AuraVoice - Señales de Detección (DetectionSignals.jsx)
 * 
 * Componente visual que muestra las señales extraídas por el modelo Wav2Vec2
 * (frecuencias, formantes, latencia) y cómo aportan al cálculo del LLR total.
 */
import React from 'react';
import {
  Waves,
  MessageSquare,
  Layers,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';


export default function DetectionSignals({
  isSynthetic,
  confidence,
  scoreTotal,
  acousticLLR,
  behavioralLLR,
  acousticSegments,
  behavioralEvents
}) {
  const synthetic = Boolean(isSynthetic);

  const confidencePct = Math.round(
    (confidence || 0) * 1000
  ) / 10;


  return (
    <div
      className="avant-card fx-card"
      style={{
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px'
      }}
    >

      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '10px'
        }}
      >
        <div>
          <span className="fx-kicker">Evidencia del modelo</span>
          <h3 className="fx-card-title">Señales del Modelo</h3>
        </div>

        <span
          className={`altur-badge ${synthetic
              ? 'badge-rose'
              : 'badge-emerald'
            }`}
        >
          {synthetic
            ? 'Voz sintética'
            : 'Voz humana'}
        </span>
      </div>


      {/* Main model evidence */}
      <div
        style={{
          background: '#f8fafc',
          border: '1px solid var(--border-card)',
          borderRadius: 'var(--radius-sm)',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '18px'
        }}
      >

        {/* Acoustic */}
        <div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '7px'
            }}
          >
            <span
              style={{
                fontSize: '0.8rem',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Waves
                size={14}
                color="#0284c7"
              />

              Evidencia acústica
            </span>

            <span
              className="font-mono"
              style={{
                fontSize: '0.75rem',
                fontWeight: 800
              }}
            >
              {acousticLLR?.toFixed(2) ?? '—'}
            </span>
          </div>

          <div
            style={{
              height: '7px',
              background: '#e2e8f0',
              borderRadius: '999px',
              overflow: 'hidden'
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${Math.min(
                  100,
                  Math.abs(acousticLLR || 0) /
                  Math.max(
                    Math.abs(scoreTotal || 1),
                    1
                  ) *
                  100
                )}%`,
                background: synthetic
                  ? 'linear-gradient(90deg, #f59e0b, #e11d48)'
                  : 'linear-gradient(90deg, #0284c7, #059669)'
              }}
            />
          </div>

          <p
            style={{
              fontSize: '0.7rem',
              color: 'var(--text-secondary)',
              marginTop: '5px'
            }}
          >
            {acousticSegments ?? 0} segmentos acústicos procesados
          </p>
        </div>


        {/* Behavioral */}
        <div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '7px'
            }}
          >
            <span
              style={{
                fontSize: '0.8rem',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <MessageSquare
                size={14}
                color="#6366f1"
              />

              Evidencia conversacional
            </span>

            <span
              className="font-mono"
              style={{
                fontSize: '0.75rem',
                fontWeight: 800
              }}
            >
              {behavioralLLR?.toFixed(2) ?? '—'}
            </span>
          </div>

          <div
            style={{
              height: '7px',
              background: '#e2e8f0',
              borderRadius: '999px',
              overflow: 'hidden'
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${Math.min(
                  100,
                  Math.abs(behavioralLLR || 0) /
                  Math.max(
                    Math.abs(scoreTotal || 1),
                    1
                  ) *
                  100
                )}%`,
                background: synthetic
                  ? 'linear-gradient(90deg, #f59e0b, #e11d48)'
                  : 'linear-gradient(90deg, #6366f1, #059669)'
              }}
            />
          </div>

          <p
            style={{
              fontSize: '0.7rem',
              color: 'var(--text-secondary)',
              marginTop: '5px'
            }}
          >
            {behavioralEvents ?? 0} eventos conversacionales procesados
          </p>
        </div>


        {/* Final score */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            paddingTop: '4px'
          }}
        >
          <span
            style={{
              fontSize: '0.8rem',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            {synthetic ? (
              <AlertTriangle
                size={15}
                color="#e11d48"
              />
            ) : (
              <CheckCircle
                size={15}
                color="#059669"
              />
            )}

            Puntuación final
          </span>

          <span
            className="font-mono"
            style={{
              fontSize: '0.82rem',
              fontWeight: 800
            }}
          >
            {scoreTotal?.toFixed(2) ?? '—'}
          </span>
        </div>
      </div>


      {/* Confidence */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '12px 14px',
          background: synthetic
            ? 'var(--accent-rose-light)'
            : 'var(--accent-emerald-light)',
          borderRadius: 'var(--radius-sm)'
        }}
      >
        <Layers
          size={16}
          color={
            synthetic
              ? '#e11d48'
              : '#059669'
          }
        />

        <span
          style={{
            fontSize: '0.77rem',
            fontWeight: 700
          }}
        >
          Confianza del modelo:
        </span>

        <span
          className="font-mono"
          style={{
            fontSize: '0.82rem',
            fontWeight: 800
          }}
        >
          {confidencePct}%
        </span>
      </div>

    </div>
  );
}