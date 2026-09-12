import { useEffect, useRef, useState } from 'react';
import {
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

const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') || 'http://localhost:8000';

function formatDuration(seconds) {
  const totalSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(totalSeconds / 60)).padStart(2, '0')}:${String(totalSeconds % 60).padStart(2, '0')}`;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Sin fecha';
  return date.toLocaleString('es-MX', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
}

function mapAudio(audio) {
  return {
    id: audio.id,
    name: audio.title,
    date: formatDate(audio.created_at),
    duration: formatDuration(audio.duration),
    audioUrl: audio.audio_url?.startsWith('http') ? audio.audio_url : `${API_BASE}${audio.audio_url}`,
  };
}

function Waveform({ playing }) {
  const bars = [35, 58, 82, 45, 68, 94, 52, 35, 73, 48, 88, 61, 38, 75, 54, 90, 45, 66, 34, 78, 51, 85, 42, 64, 36, 72, 50, 83, 43, 61];
  return <div className={`review-waveform ${playing ? 'is-playing' : ''}`} aria-hidden="true">
    {bars.map((height, index) => <i key={index} style={{ '--bar-height': `${height}%`, '--bar-delay': `${index * 0.025}s` }} />)}
  </div>;
}

function NoteRow({ note, playing, onPlay, onClassify }) {
  return <article className="review-note-row">
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
  const [notes, setNotes] = useState([]);
  const [stats, setStats] = useState({ pending: 0, reviewed: 0, total: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);
  const [playingId, setPlayingId] = useState(null);
  const [isTrainingOpen, setIsTrainingOpen] = useState(false);
  const [modelName, setModelName] = useState('');
  const inputRef = useRef(null);
  const audioRef = useRef(null);
  useEffect(() => {
    let isMounted = true;

    Promise.all([
      fetch(`${API_BASE}/review/audios`),
      fetch(`${API_BASE}/review/stats`),
    ])
      .then(async ([audiosResponse, statsResponse]) => {
        if (!audiosResponse.ok || !statsResponse.ok) throw new Error('No se pudieron cargar las notas.');
        return Promise.all([audiosResponse.json(), statsResponse.json()]);
      })
      .then(([audios, reviewStats]) => {
        if (!isMounted) return;
        setNotes(audios.map(mapAudio));
        setStats(reviewStats);
      })
      .catch((requestError) => {
        if (isMounted) setError(requestError.message);
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => { isMounted = false; };
  }, []);

  useEffect(() => {
    if (isTrainingOpen) inputRef.current?.focus();
  }, [isTrainingOpen]);

  useEffect(() => () => audioRef.current?.pause(), []);

  const playAudio = async (note) => {
    if (playingId === note.id) {
      audioRef.current?.pause();
      setPlayingId(null);
      return;
    }

    audioRef.current?.pause();
    const audio = new Audio(note.audioUrl);
    audio.addEventListener('ended', () => setPlayingId(null));
    audioRef.current = audio;
    try {
      await audio.play();
      setPlayingId(note.id);
      setError('');
    } catch {
      setError(`No se pudo reproducir ${note.name}.`);
      setPlayingId(null);
    }
  };

  const classify = async (id, classification) => {
    const endpoint = classification === 'Eliminar'
      ? `${API_BASE}/review/audios/${id}`
      : `${API_BASE}/review/audios/${id}/classification`;
    const options = classification === 'Eliminar'
      ? { method: 'DELETE' }
      : { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ classification: classification === 'Sintético' ? 'synthetic' : 'real' }) };

    try {
      const response = await fetch(endpoint, options);
      if (!response.ok) throw new Error('No se pudo guardar la revisión.');
      setNotes((current) => current.filter((note) => note.id !== id));
      setStats((current) => classification === 'Eliminar'
        ? { ...current, pending: Math.max(0, current.pending - 1), total: Math.max(0, current.total - 1) }
        : { ...current, pending: Math.max(0, current.pending - 1), reviewed: current.reviewed + 1 });
      setPlayingId((current) => current === id ? null : current);
      setError('');
    } catch (requestError) {
      setError(requestError.message);
    }
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
      <section className="review-metrics" aria-label="Resumen de notas"><MetricCard label="Total de grabaciones" value={stats.total} detail="Disponibles para entrenar" tone="green" /><MetricCard label="Muestras clasificadas" value={stats.reviewed} detail="Listas para el modelo" tone="blue" /><MetricCard label="Pendientes de revisión" value={notes.length} detail="Necesitan tu atención" tone="orange" /></section>
      <section className="review-model-card"><div className="review-model-header"><div><p className="review-card-kicker">Datos para entrenamiento</p><h2>Vocalis / base-01</h2></div><span className="review-model-state">{stats.pending} pendientes</span></div><div className="review-model-log"><p><time>PENDIENTES</time><span>{stats.pending} grabaciones esperan revisión</span></p><p><time>CLASIFICADAS</time><span>{stats.reviewed} muestras listas para entrenar</span></p><p><time>TOTAL</time><span>{stats.total} grabaciones disponibles</span></p></div></section>
      <section className="review-list-section"><button className="review-section-heading" onClick={() => setIsExpanded((current) => !current)} aria-expanded={isExpanded}><span><span className="review-section-icon"><Headphones size={18} /></span><span><strong>Revisar notas</strong><small>{isLoading ? 'Cargando notas...' : `${notes.length} notas esperan tu revisión`}</small></span></span><ChevronDown className={isExpanded ? 'rotate' : ''} size={21} /></button>{isExpanded && <div className="review-notes-list">{error && <p className="review-error-message">{error}</p>}{isLoading ? <div className="review-empty-state"><span>Cargando notas...</span></div> : notes.length ? notes.map((note) => <NoteRow key={note.id} note={note} playing={playingId === note.id} onPlay={() => playAudio(note)} onClassify={(classification) => classify(note.id, classification)} />) : <div className="review-empty-state"><Check size={20} /><strong>Todo revisado</strong><span>Ya clasificaste todas las notas de esta sesión.</span></div>}</div>}</section>
    </div>
    <p className="review-privacy-note"><Mic2 size={14} />Tus revisiones ayudan a mejorar el modelo. Los audios se procesan de forma privada.</p>
    {isTrainingOpen && <div className="review-modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setIsTrainingOpen(false); }}><section className="review-training-modal" role="dialog" aria-modal="true" aria-labelledby="review-training-title"><button className="review-close-button" onClick={() => setIsTrainingOpen(false)} aria-label="Cerrar"><X size={18} /></button><div className="review-modal-symbol"><Sparkles size={22} /></div><p className="review-eyebrow">NUEVA VERSIÓN</p><h2 id="review-training-title">Ingresa el nombre<br />del nuevo modelo</h2><p className="review-modal-copy">Dale un nombre para reconocerlo fácilmente cuando esté listo.</p><form onSubmit={submitTraining}><label htmlFor="review-model-name">Nombre del modelo</label><div className="review-input-wrap"><input id="review-model-name" ref={inputRef} value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="Ej. Vocalis primavera" /><span><CornerDownLeft size={13} />Enter</span></div><button className="review-primary-button review-modal-submit" type="submit">Comenzar entrenamiento <Sparkles size={16} /></button></form></section></div>}
  </div>;
}
