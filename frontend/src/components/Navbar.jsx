import React from 'react';
import ChameleonLogo from './ChameleonLogo';
import { PhoneCall, FileAudio } from 'lucide-react';

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
        <div className="segmented-track" style={{ maxWidth: '440px', width: '100%' }}>
          <button
            className={`segmented-btn ${activeMode === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveMode('batch')}
          >
            <FileAudio size={16} />
            <span>Audio Forensics</span>
            <span className="font-mono" style={{ fontSize: '0.7rem', opacity: 0.65 }}>/detect</span>
          </button>

          <button
            className={`segmented-btn ${activeMode === 'streaming' ? 'active' : ''}`}
            onClick={() => setActiveMode('streaming')}
          >
            <PhoneCall size={16} />
            <span>Live Call</span>
            <span className="font-mono" style={{ fontSize: '0.7rem', opacity: 0.65 }}>/detect_streaming</span>
          </button>
        </div>

        {/* Status Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div className="altur-badge badge-emerald" style={{ padding: '6px 14px' }}>
            <span className="dot-pulse green"></span>
            <span>Sistema Altur Operativo</span>
          </div>
        </div>
      </div>
    </header>
  );
}
