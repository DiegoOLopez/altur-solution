import { useEffect, useRef, useState } from 'react';
import {
  AudioLines,
  Check,
  ChevronDown,
  CornerDownLeft,
  Headphones,
  Mic2,
  Pause,
  Play,
  Sparkles,
  Trash2,
  X
} from 'lucide-react';

const initialNotes = [
  { id: 1, name: 'Llamada de bienvenida', date: 'Hoy, 10:42', duration: '02:18', accent: 'mint' },
  { id: 2, name: 'Consulta sobre el pedido', date: 'Hoy, 09:18', duration: '01:46', accent: 'sky' },
  { id: 3, name: 'Seguimiento de suscripción', date: 'Ayer, 16:27', duration: '03:04', accent: 'lilac' },
  { id: 4, name: 'Cambio de dirección', date: 'Ayer, 14:03', duration: '00:58', accent: 'peach' }
];

function Waveform({ playing }) {
  const bars = [35, 58, 82, 45, 68, 94, 52, 35, 73, 48, 88, 61, 38, 75, 54, 90, 45, 66, 34, 78, 51, 85, 42, 64, 36, 72, 50, 83, 43, 61];
  return <div className={`review-waveform ${playing ? 'is-playing' : ''}`} aria-hidden="true">
    {bars.map((height, index) => <i key={index} style={{ '--bar-height': `${height}%`, '--bar-delay': `${index * 0.025}s` }} />)}
  </div>;
}

function NoteRow({ note, playing, onPlay, onClassify }) {
  return <article className="review-note-row">
    <div className={`review-note-art ${note.accent}`}><AudioLines size={20} /></div>
    <div className="review-note-info"><h3>{note.name}</h3><p>{note.date}<span>•</span>{note.duration}</p></div>
    <div className="review-note-player"><button className="review-play-button" onClick={onPlay} aria-label={playing ? `Pausar ${note.name}` : `Reproducir ${note.name}`}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><Waveform playing={playing} /></div>
    <div className="review-note-actions" role="group" aria-label={`Clasificar ${note.name}`}>
      <button onClick={() => onClassify('Sintético')}>Sintético</button>
      <button onClick={() => onClassify('Real')}>Real</button>
      <button className="delete-action" onClick={() => onClassify('Eliminar')} aria-label="Eliminar nota"><Trash2 size={14} /></button>
    </div>
  </article>;
}

function MetricCard({ label, value, detail, tone }) {
  return <article className={`review-metric-card ${tone}`}><div><p>{label}</p><strong>{value}</strong><span>{detail}</span></div><div className="review-metric-rule"><i /></div></article>;
}

export default function ReviewHub() {
  const [notes, setNotes] = useState(initialNotes);
  const [isExpanded, setIsExpanded] = useState(true);
  const [playingId, setPlayingId] = useState(null);
  const [isTrainingOpen, setIsTrainingOpen] = useState(false);
  const [modelName, setModelName] = useState('');
  const inputRef = useRef(null);
  const reviewedCount = initialNotes.length - notes.length;

  useEffect(() => {
    if (isTrainingOpen) inputRef.current?.focus();
  }, [isTrainingOpen]);

  const classify = (id) => {
    setNotes((current) => current.filter((note) => note.id !== id));
    setPlayingId((current) => current === id ? null : current);
  };

  const submitTraining = (event) => {
    event.preventDefault();
    if (!modelName.trim()) return;
    setIsTrainingOpen(false);
    setModelName('');
  };

  return <div className="review-hub">
    <section className="review-intro"><div><p className="review-eyebrow"><span />Centro de revisión</p><h1>Haz que cada voz<br /><em>cuente.</em></h1><p className="review-intro-copy">Revisa tus notas de audio y ayuda a Vocalis a entender mejor las conversaciones de tu equipo.</p></div><button className="review-primary-button" onClick={() => setIsTrainingOpen(true)}><Sparkles size={18} />Entrenar de nuevo</button></section>
    <div className="review-bento-grid">
      <section className="review-metrics" aria-label="Resumen de notas"><MetricCard label="Notas nuevas totales" value={initialNotes.length} detail="En esta sesión" tone="green" /><MetricCard label="Notas revisadas" value={reviewedCount} detail="Clasificadas por ti" tone="blue" /><MetricCard label="Notas sin revisar" value={notes.length} detail="Necesitan tu atención" tone="orange" /></section>
      <section className="review-model-card"><div className="review-model-header"><div><p className="review-card-kicker">Modelo en entrenamiento</p><h2>Vocalis / base-01</h2></div><span className="review-model-state">Listo para datos</span></div><div className="review-model-log"><p><time>10:42:18</time><span>4 notas nuevas detectadas</span></p><p><time>10:43:02</time><span>{reviewedCount} muestras clasificadas</span></p><p><time>10:43:19</time><span>Esperando tu siguiente lote</span></p></div></section>
      <section className="review-list-section"><button className="review-section-heading" onClick={() => setIsExpanded((current) => !current)} aria-expanded={isExpanded}><span><span className="review-section-icon"><Headphones size={18} /></span><span><strong>Revisar notas</strong><small>{notes.length} notas esperan tu revisión</small></span></span><ChevronDown className={isExpanded ? 'rotate' : ''} size={21} /></button>{isExpanded && <div className="review-notes-list">{notes.length ? notes.map((note) => <NoteRow key={note.id} note={note} playing={playingId === note.id} onPlay={() => setPlayingId(playingId === note.id ? null : note.id)} onClassify={() => classify(note.id)} />) : <div className="review-empty-state"><Check size={20} /><strong>Todo revisado</strong><span>Ya clasificaste todas las notas de esta sesión.</span></div>}</div>}</section>
    </div>
    <p className="review-privacy-note"><Mic2 size={14} />Tus revisiones ayudan a mejorar el modelo. Los audios se procesan de forma privada.</p>
    {isTrainingOpen && <div className="review-modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setIsTrainingOpen(false); }}><section className="review-training-modal" role="dialog" aria-modal="true" aria-labelledby="review-training-title"><button className="review-close-button" onClick={() => setIsTrainingOpen(false)} aria-label="Cerrar"><X size={18} /></button><div className="review-modal-symbol"><Sparkles size={22} /></div><p className="review-eyebrow">NUEVA VERSIÓN</p><h2 id="review-training-title">Ingresa el nombre<br />del nuevo modelo</h2><p className="review-modal-copy">Dale un nombre para reconocerlo fácilmente cuando esté listo.</p><form onSubmit={submitTraining}><label htmlFor="review-model-name">Nombre del modelo</label><div className="review-input-wrap"><input id="review-model-name" ref={inputRef} value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="Ej. Vocalis primavera" /><span><CornerDownLeft size={13} />Enter</span></div><button className="review-primary-button review-modal-submit" type="submit">Comenzar entrenamiento <Sparkles size={16} /></button></form></section></div>}
  </div>;
}
