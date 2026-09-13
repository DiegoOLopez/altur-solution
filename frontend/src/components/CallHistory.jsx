import React from 'react';
import { ShieldAlert, ShieldCheck } from 'lucide-react';

export default function CallHistory({ history, onSelectHistoryItem }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="avant-card fx-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
        <div>
          <span className="fx-kicker">Historial de auditorías</span>
          <h3 className="fx-card-title">Registro de Auditorías Telefónicas Recientes</h3>
        </div>
        <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
          {history.length} llamadas analizadas
        </span>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '12px'
      }}>
        {history.map((item, idx) => (
          <div
            key={idx}
            onClick={() => onSelectHistoryItem(item)}
            style={{
              background: '#f8fafc',
              border: `1px solid ${item.is_synthetic ? 'rgba(225, 29, 72, 0.25)' : 'rgba(5, 150, 105, 0.25)'}`,
              borderRadius: 'var(--radius-sm)',
              padding: '12px 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#ffffff';
              e.currentTarget.style.boxShadow = 'var(--shadow-md)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#f8fafc';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: item.is_synthetic ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {item.is_synthetic ? (
                  <ShieldAlert size={18} color="#e11d48" />
                ) : (
                  <ShieldCheck size={18} color="#059669" />
                )}
              </div>
              <div>
                <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--altur-black)' }}>
                  {item.title || `Llamada #${idx + 1}`}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {item.timestamp} • {item.duration?.toFixed(1)}s
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div className="font-mono" style={{
                fontSize: '0.9rem',
                fontWeight: 800,
                color: item.is_synthetic ? '#e11d48' : '#059669'
              }}>
                {(item.confidence * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {item.latency_ms}ms
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
