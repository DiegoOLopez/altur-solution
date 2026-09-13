/**
 * AuraVoice - Barra de Navegación (Navbar.jsx)
 * 
 * Menú principal que permite alternar entre las tres vistas de la aplicación:
 * - Detección Forense (Batch)
 * - Simulación en Vivo (Streaming)
 * - Centro de Revisión (Review Hub)
 */
import React from 'react';
import ChameleonLogo from './ChameleonLogo';
import { PhoneCall, FileAudio, ListChecks } from 'lucide-react';

export default function Navbar({ activeMode, setActiveMode }) {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-card)',
      background: 'rgba(255, 255, 255, 0.88)',
      backdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '14px 28px'
    }}>
      <div style={{
        maxWidth: '1440px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Brand */}
        <ChameleonLogo size={40} showWordmark={true} />

        {/* High-End Segmented Switcher for Both Endpoints */}
        <div className="segmented-track" style={{ maxWidth: '620px', width: '100%' }}>
          <button
            className={`segmented-btn ${activeMode === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveMode('batch')}
          >
            <FileAudio size={16} />
            <span>Análisis Forense</span>
          </button>

          <button
            className={`segmented-btn ${activeMode === 'review' ? 'active' : ''}`}
            onClick={() => setActiveMode('review')}
          >
            <ListChecks size={16} />
            <span>Revisar notas</span>
          </button>

          <button
            className={`segmented-btn ${activeMode === 'streaming' ? 'active' : ''}`}
            onClick={() => setActiveMode('streaming')}
          >
            <PhoneCall size={16} />
            <span>Llamada en Vivo</span>
          </button>
        </div>
      </div>
    </header>
  );
}
