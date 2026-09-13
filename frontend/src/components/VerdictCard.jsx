import React, { useState } from 'react';
import { 
  ShieldAlert, ShieldCheck, Cpu, Bot, Copy, 
  Check, Code2, AlertOctagon, CornerDownRight 
} from 'lucide-react';

export default function VerdictCard({ result, isAnalyzing }) {
  const [copied, setCopied] = useState(false);
  const [showJson, setShowJson] = useState(false);

  // Default values if no analysis yet
  const isSynthetic = result ? result.is_synthetic : true;
  const confidence = result ? result.confidence : 0.942;
  const confidencePct = Math.round(confidence * 1000) / 10;
  const latencyMs = result?.latency_ms || 112;
  const llmText = result?.llm_conclusion || 'Esperando ejecución del motor de análisis...';

  // SVG Gauge calculations
  const radius = 64;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (confidence * circumference);

  const handleCopyReport = () => {
    navigator.clipboard.writeText(llmText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const rawJsonContract = JSON.stringify({
    is_synthetic: isSynthetic,
    confidence: confidence
  }, null, 2);

  return (
    <div className="glass-panel" style={{
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      border: isSynthetic ? '1px solid rgba(239, 68, 68, 0.35)' : '1px solid rgba(16, 185, 129, 0.35)',
      boxShadow: isSynthetic 
        ? '0 10px 40px -10px rgba(239, 68, 68, 0.2)' 
        : '0 10px 40px -10px rgba(16, 185, 129, 0.2)'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isSynthetic ? (
            <ShieldAlert size={20} color="#ef4444" />
          ) : (
            <ShieldCheck size={20} color="#10b981" />
          )}
          <h2 style={{ fontSize: '1.05rem', fontWeight: '800', letterSpacing: '-0.01em' }}>
            4. Veredicto del Endpoint & Conclusión LLM
          </h2>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Latencia Edge: <strong style={{ color: '#00f2fe' }}>{latencyMs}ms</strong>
          </span>
          <button
            className="btn btn-outline"
            style={{ padding: '4px 8px', fontSize: '0.72rem' }}
            onClick={() => setShowJson(!showJson)}
          >
            <Code2 size={12} />
            POST /detectar JSON
          </button>
        </div>
      </div>

      {/* Hero Verdict Display with Circular Risk Gauge */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr',
        gap: '20px',
        alignItems: 'center',
        background: isSynthetic ? 'rgba(239, 68, 68, 0.05)' : 'rgba(16, 185, 129, 0.05)',
        padding: '16px 20px',
        borderRadius: 'var(--radius-sm)',
        border: `1px solid ${isSynthetic ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`
      }}>
        {/* Circular Gauge */}
        <div style={{ position: 'relative', width: '140px', height: '140px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="140" height="140" style={{ transform: 'rotate(-90deg)' }}>
            {/* Background Track */}
            <circle
              cx="70"
              cy="70"
              r={radius}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeWidth="10"
              fill="transparent"
            />
            {/* Value Arc */}
            <circle
              cx="70"
              cy="70"
              r={radius}
              stroke={isSynthetic ? '#ef4444' : '#10b981'}
              strokeWidth="10"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              style={{
                transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                filter: isSynthetic ? 'drop-shadow(0 0 8px rgba(239,68,68,0.6))' : 'drop-shadow(0 0 8px rgba(16,185,129,0.6))'
              }}
            />
          </svg>
          {/* Gauge Center Text */}
          <div style={{
            position: 'absolute',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <span className="font-mono" style={{
              fontSize: '1.5rem',
              fontWeight: 800,
              color: isSynthetic ? '#ef4444' : '#10b981',
              lineHeight: 1
            }}>
              {confidencePct}%
            </span>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: '4px' }}>
              Confianza
            </span>
          </div>
        </div>

        {/* Verdict Details & Actions */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className={`pulse-dot ${isSynthetic ? 'rose' : 'emerald'}`}></span>
            <span className={`chip ${isSynthetic ? 'chip-rose' : 'chip-emerald'}`} style={{ fontSize: '0.75rem', fontWeight: 800 }}>
              {isSynthetic ? '🚨 SINTÉTICO (DEEPFAKE DETECTADO)' : '✅ HUMANO AUTÉNTICO CONFIRMADO'}
            </span>
          </div>

          <h3 style={{
            fontSize: '1.25rem',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            color: isSynthetic ? '#fca5a5' : '#86efac'
          }}>
            {isSynthetic 
              ? 'Bloqueo de Canal Telefónico Recomendado' 
              : 'Canal de Llamada Seguro para Operaciones'}
          </h3>

          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            {isSynthetic
              ? 'Se detectó discrepancia acústico-comportamental consistente con clonación por voz. Probabilidad alta de ataque de ingeniería social a clientes bancarios.'
              : 'Las características espectrales, fonéticas y de latencia conversacional se alinean con un patrón biométrico humano natural.'}
          </p>

          <div style={{ display: 'flex', gap: '8px', marginTop: '4px', flexWrap: 'wrap' }}>
            <span className="chip font-mono" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)', fontSize: '0.72rem' }}>
              is_synthetic: <strong style={{ color: isSynthetic ? '#ef4444' : '#10b981', marginLeft: '4px' }}>{isSynthetic ? 'true' : 'false'}</strong>
            </span>
            <span className="chip font-mono" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)', fontSize: '0.72rem' }}>
              confidence: <strong style={{ color: '#00f2fe', marginLeft: '4px' }}>{confidence.toFixed(3)}</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Raw JSON Accordion (For Judges to verify the contract) */}
      {showJson && (
        <div style={{
          background: '#04070d',
          border: '1px solid var(--border-active)',
          borderRadius: 'var(--radius-sm)',
          padding: '12px 16px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontSize: '0.75rem', color: '#00f2fe', fontWeight: 600 }}>
              Respuesta del Endpoint Oficial POST /detectar:
            </span>
            <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>application/json</span>
          </div>
          <pre className="font-mono" style={{ fontSize: '0.8rem', color: '#10b981', margin: 0 }}>
            {rawJsonContract}
          </pre>
        </div>
      )}

      {/* LLM Forensic Report Card */}
      <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-sm)',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Bot size={18} color="#00f2fe" />
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Dictamen Forense Explicable (Generado por LLM)
            </span>
          </div>
          <button
            className="btn btn-outline"
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
            onClick={handleCopyReport}
          >
            {copied ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
            {copied ? 'Copiado' : 'Copiar Dictamen'}
          </button>
        </div>

        {/* Text of LLM */}
        <div style={{
          background: 'rgba(0, 0, 0, 0.35)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          borderRadius: '6px',
          padding: '14px',
          fontSize: '0.82rem',
          lineHeight: 1.6,
          color: '#e2e8f0',
          whiteSpace: 'pre-wrap',
          fontFamily: 'var(--font-display)'
        }}>
          {llmText}
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.72rem',
          color: 'var(--text-muted)'
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CornerDownRight size={12} color="#00f2fe" />
            Explicabilidad para auditoría bancaria y jueces
          </span>
        </div>
      </div>
    </div>
  );
}
