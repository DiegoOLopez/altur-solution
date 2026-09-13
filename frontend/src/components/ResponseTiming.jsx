import React from 'react';
import {
  MessageSquare,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';


export default function ResponseTiming({
  isSynthetic,
  behavioralLLR,
  behavioralEvents,
  scoreTotal
}) {
  const synthetic = Boolean(isSynthetic);

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
          <span className="fx-kicker">Comportamiento conversacional</span>
          <h3 className="fx-card-title">Evidencia Conversacional</h3>
        </div>

        <span
          className={`altur-badge ${synthetic
              ? 'badge-rose'
              : 'badge-emerald'
            }`}
        >
          {synthetic
            ? 'Contribución sintética'
            : 'Contribución humana'}
        </span>
      </div>


      {/* Explanation */}
      <p
        style={{
          fontSize: '0.76rem',
          color: 'var(--text-secondary)',
          lineHeight: 1.5
        }}
      >
        El modelo analiza eventos conversacionales entre el
        llamante y el agente. Esta vista muestra la evidencia
        comportamental acumulada por el detector.
      </p>


      {/* Behavioral evidence */}
      <div
        style={{
          background: '#f8fafc',
          border: '1px solid var(--border-card)',
          borderRadius: 'var(--radius-sm)',
          padding: '18px'
        }}
      >

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '14px'
          }}
        >
          <MessageSquare
            size={17}
            color="#6366f1"
          />

          <span
            style={{
              fontSize: '0.85rem',
              fontWeight: 800
            }}
          >
            LLR comportamental acumulado
          </span>
        </div>


        <div
          className="font-mono"
          style={{
            fontSize: '1.7rem',
            fontWeight: 800,
            marginBottom: '5px'
          }}
        >
          {behavioralLLR?.toFixed(2) ?? '—'}
        </div>


        <p
          style={{
            fontSize: '0.72rem',
            color: 'var(--text-secondary)'
          }}
        >
          Evidencia aportada por el bloque comportamental
          del modelo.
        </p>
      </div>


      {/* Event count */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '12px'
        }}
      >

        <div
          style={{
            background: '#f8fafc',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '14px'
          }}
        >
          <div
            style={{
              fontSize: '0.72rem',
              color: 'var(--text-secondary)',
              marginBottom: '5px'
            }}
          >
            Eventos analizados
          </div>

          <div
            className="font-mono"
            style={{
              fontSize: '1.15rem',
              fontWeight: 800
            }}
          >
            {behavioralEvents ?? 0}
          </div>
        </div>


        <div
          style={{
            background: '#f8fafc',
            border: '1px solid var(--border-card)',
            borderRadius: 'var(--radius-sm)',
            padding: '14px'
          }}
        >
          <div
            style={{
              fontSize: '0.72rem',
              color: 'var(--text-secondary)',
              marginBottom: '5px'
            }}
          >
            Score total
          </div>

          <div
            className="font-mono"
            style={{
              fontSize: '1.15rem',
              fontWeight: 800
            }}
          >
            {scoreTotal?.toFixed(2) ?? '—'}
          </div>
        </div>

      </div>


      {/* Interpretation */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '9px',
          background: synthetic
            ? 'var(--accent-rose-light)'
            : 'var(--accent-emerald-light)',
          border: `1px solid ${synthetic
              ? 'rgba(225, 29, 72, 0.2)'
              : 'rgba(5, 150, 105, 0.2)'
            }`,
          borderRadius: 'var(--radius-sm)',
          padding: '12px 14px'
        }}
      >

        {synthetic ? (
          <AlertTriangle
            size={16}
            color="#e11d48"
          />
        ) : (
          <CheckCircle
            size={16}
            color="#059669"
          />
        )}

        <span
          style={{
            fontSize: '0.77rem',
            color: 'var(--text-primary)',
            lineHeight: 1.4,
            fontWeight: 600
          }}
        >
          {synthetic
            ? 'La evidencia comportamental contribuye al score del modelo hacia la hipótesis de voz sintética.'
            : 'La evidencia comportamental contribuye al score del modelo hacia la hipótesis de voz humana.'}
        </span>

      </div>

    </div>
  );
}