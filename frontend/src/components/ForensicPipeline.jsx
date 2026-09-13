/**
 * AuraVoice - Tubería Forense (ForensicPipeline.jsx)
 * 
 * Componente visual que ilustra los pasos sucesivos del proceso de ingesta
 * (validación de formato, resampleo) y análisis de inferencia del modelo.
 */
import React from 'react';
import { CheckCircle2, Circle, Loader2, XCircle, Radar, Terminal } from 'lucide-react';

const STATUS_META = {
  done: { Icon: CheckCircle2, color: '#059669', label: 'Hecho', bg: 'var(--accent-emerald-light)' },
  active: { Icon: Loader2, color: '#0284c7', label: 'Procesando', bg: 'var(--accent-cyan-light)', spin: true },
  error: { Icon: XCircle, color: '#e11d48', label: 'Rechazado', bg: 'var(--accent-rose-light)' },
  pending: { Icon: Circle, color: '#cbd5e1', label: 'Pendiente', bg: '#f1f5f9' }
};

export default function ForensicPipeline({ title, steps, badge, live }) {
  const hasActive = (steps || []).some((s) => s.status === 'active');

  return (
    <div className="avant-card fx-card" style={{
      padding: '18px 20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {hasActive ? (
            <Radar size={17} color="#0284c7" />
          ) : (
            <Terminal size={17} color="#64748b" />
          )}
          <h4 style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--altur-black)', letterSpacing: '-0.01em' }}>
            {title}
          </h4>
        </div>
        {badge && (
          <span className={`altur-badge font-mono ${live ? 'badge-cyan' : ''}`} style={live ? undefined : { background: '#e2e8f0', color: '#64748b', fontSize: '0.68rem' }}>
            {badge}
          </span>
        )}
      </div>

      {/* Steps */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {(steps || []).map((step, idx) => {
          const meta = STATUS_META[step.status] || STATUS_META.pending;
          const { Icon } = meta;
          const isLast = idx === steps.length - 1;

          return (
            <div key={step.id || idx} style={{ display: 'flex', gap: '12px' }}>
              {/* Icon column + connector */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div style={{
                  width: '30px',
                  height: '30px',
                  borderRadius: '50%',
                  background: meta.bg,
                  border: `1.5px solid ${step.status === 'active' ? '#0284c7' : meta.color}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  <Icon
                    size={15}
                    color={meta.color}
                    className={meta.spin ? 'rotate-spin' : ''}
                  />
                </div>
                {!isLast && (
                  <div style={{
                    width: '2px',
                    flex: 1,
                    minHeight: '16px',
                    background: step.status === 'done' ? 'rgba(5, 150, 105, 0.35)' : '#e2e8f0',
                    margin: '2px 0'
                  }} />
                )}
              </div>

              {/* Content */}
              <div style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '10px',
                paddingBottom: isLast ? 0 : '14px'
              }}>
                <div>
                  <div style={{
                    fontSize: '0.83rem',
                    fontWeight: 700,
                    color: step.status === 'pending' ? 'var(--text-muted)' : 'var(--altur-black)'
                  }}>
                    {step.label}
                  </div>
                  {step.sublabel && (
                    <div className="font-mono" style={{ fontSize: '0.68rem', color: 'var(--text-faint)', marginTop: '1px' }}>
                      {step.sublabel}
                    </div>
                  )}
                </div>
                <span className={`altur-badge font-mono`} style={{
                  background: meta.bg,
                  color: meta.color,
                  fontSize: '0.62rem',
                  padding: '2px 9px',
                  flexShrink: 0
                }}>
                  {meta.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}