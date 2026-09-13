/**
 * AuraVoice - Centro de Revisión (ReviewHub.jsx)
 * 
 * Permite a los agentes humanos auditar los audios procesados previamente.
 * Se comunica con la base de datos MySQL (vía los endpoints /review/*)
 * para listar, reproducir y clasificar grabaciones, así como lanzar el
 * reentrenamiento automático del modelo con los nuevos audios clasificados.
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
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
    initialClassification: audio.is_synthetic ? 'Sintético' : 'Humano',
    audioUrl: audio.audio_url?.startsWith('http') ? audio.audio_url : `${API_BASE}${audio.audio_url}`,
  };
}

function Waveform({ playing }) {
  const bars = [35, 58, 82, 45, 68, 94, 52, 35, 73, 48, 88, 61, 38, 75, 54, 90, 45, 66, 34, 78, 51, 85, 42, 64, 36, 72, 50, 83, 43, 61];
  return <div className={`review-waveform ${playing ? 'is-playing' : ''}`} aria-hidden="true">
    {bars.map((height, index) => <i key={index} style={{ '--bar-height': `${height}%`, '--bar-delay': `${index * 0.025}s` }} />)}
  </div>;
}

function NoteRow({ note, playing, selectedClassification, onPlay, onSelect, onConfirm, isSaving }) {
  return <article className="review-note-row">
    <div className="review-note-info"><h3>{note.name}</h3><p>{note.date}<span>•</span>{note.duration}</p></div>
    <div className="review-note-player"><button className="review-play-button" onClick={onPlay} aria-label={playing ? `Pausar ${note.name}` : `Reproducir ${note.name}`}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><Waveform playing={playing} /></div>
    <div className="review-note-actions" role="group" aria-label={`Clasificar ${note.name}`}>
      <button className={selectedClassification === 'Sintético' ? 'is-selected' : ''} onClick={() => onSelect('Sintético')}>Sintético</button>
      <button className={selectedClassification === 'Humano' ? 'is-selected' : ''} onClick={() => onSelect('Humano')}>Humano</button>
      <button className={`delete-action ${selectedClassification === 'Eliminar' ? 'is-selected' : ''}`} onClick={() => onSelect('Eliminar')} aria-label="Eliminar nota"><Trash2 size={14} /></button>
      <button className="confirm-action" onClick={onConfirm} disabled={isSaving} aria-label={`Confirmar selección para ${note.name}`}><Check size={14} />{isSaving ? 'Guardando...' : 'Confirmar'}</button>
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
  const [selections, setSelections] = useState({});
  const [savingId, setSavingId] = useState(null);
  const [isExpanded, setIsExpanded] = useState(true);
  const [playingId, setPlayingId] = useState(null);
  const [isTrainingOpen, setIsTrainingOpen] = useState(false);
  const [modelName, setModelName] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [isCancellingTraining, setIsCancellingTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState({ status: 'idle', progress: 0, step: 'Sin entrenamiento activo' });
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
        const mappedAudios = audios.map(mapAudio);
        setNotes(mappedAudios);
        setSelections(Object.fromEntries(mappedAudios.map((audio) => [audio.id, audio.initialClassification])));
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

  useEffect(() => {
    let isMounted = true;
    let intervalId;

    const refreshTrainingStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/review/train/status`);
        if (!response.ok) return;
        const status = await response.json();
        if (!isMounted) return;
        setTrainingStatus(status);
      } catch {
        // The review API error is already surfaced by the main data request.
      }
    };

    refreshTrainingStatus();
  intervalId = window.setInterval(refreshTrainingStatus, 2000);
    return () => {
      isMounted = false;
      if (intervalId) window.clearInterval(intervalId);
    };
  }, []);

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

  const confirmClassification = async (id) => {
    const classification = selections[id];
    if (!classification) return;

    const endpoint = classification === 'Eliminar'
      ? `${API_BASE}/review/audios/${id}`
      : `${API_BASE}/review/audios/${id}/classification`;
    const options = classification === 'Eliminar'
      ? { method: 'DELETE' }
      : { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ classification: classification === 'Sintético' ? 'synthetic' : 'real' }) };

    setSavingId(id);
    try {
      const response = await fetch(endpoint, options);
      if (!response.ok) throw new Error('No se pudo guardar la revisión.');
      setNotes((current) => current.filter((note) => note.id !== id));
      setSelections((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      setStats((current) => classification === 'Eliminar'
        ? { ...current, pending: Math.max(0, current.pending - 1), total: Math.max(0, current.total - 1) }
        : { ...current, pending: Math.max(0, current.pending - 1), reviewed: current.reviewed + 1 });
      setPlayingId((current) => current === id ? null : current);
      setError('');
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSavingId(null);
    }
  };

  const submitTraining = async (event) => {
    event.preventDefault();
    if (!modelName.trim() || isTraining) return;

    setIsTraining(true);
    setTrainingResult(null);
    setError('');

    try {
      const response = await fetch(`${API_BASE}/review/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Error durante el entrenamiento.');
      }

      setTrainingStatus({ status: 'training', progress: 0, step: 'Preparando datos', model_name: data.model_name });
      closeTrainingModal();
    } catch (trainError) {
      setError(trainError.message);
      setTrainingResult(null);
    } finally {
      setIsTraining(false);
    }
  };

  const closeTrainingModal = () => {
    setIsTrainingOpen(false);
    setModelName('');
    setTrainingResult(null);
  };

  const cancelTraining = async () => {
    if (isCancellingTraining) return;

    setIsCancellingTraining(true);
    try {
      const response = await fetch(`${API_BASE}/review/train/cancel`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'No se pudo cancelar el entrenamiento.');
      setTrainingStatus(data);
      setError('');
    } catch (cancelError) {
      setError(cancelError.message);
    } finally {
      setIsCancellingTraining(false);
    }
  };

  return <div className="review-hub">
    <section className="review-intro"><div><p className="review-eyebrow"><span />Centro de revisión</p><h1>Haz que cada voz<br /><em>cuente.</em></h1><p className="review-intro-copy">Revisa tus notas de audio y ayuda a Vocalis a entender mejor las conversaciones de tu equipo.</p></div><button className="review-primary-button" onClick={() => setIsTrainingOpen(true)} disabled={trainingStatus.status === 'training' || isTraining} title={trainingStatus.status === 'training' ? 'Ya hay un entrenamiento en curso' : undefined}><Sparkles size={18} />{trainingStatus.status === 'training' ? 'Entrenamiento en curso' : 'Entrenar de nuevo'}</button></section>
    <div className="review-bento-grid">
      <section className="review-metrics" aria-label="Resumen de notas"><MetricCard label="Total de grabaciones" value={stats.total} detail="Disponibles para entrenar" tone="green" /><MetricCard label="Muestras clasificadas" value={stats.reviewed} detail="Listas para el modelo" tone="blue" /><MetricCard label="Pendientes de revisión" value={notes.length} detail="Necesitan tu atención" tone="orange" /></section>
      <section className="review-model-card"><div className="review-model-header"><div><p className="review-card-kicker">Datos para entrenamiento</p><h2>{trainingStatus.model_name || 'Vocalis / base-01'}</h2></div><span className="review-model-state">{trainingStatus.status === 'training' ? `${trainingStatus.progress}% en curso` : `${stats.pending} pendientes`}</span></div><div className="review-model-log"><p><time>PENDIENTES</time><span>{stats.pending} grabaciones esperan revisión</span></p><p><time>CLASIFICADAS</time><span>{stats.reviewed} muestras listas para entrenar</span></p><p><time>TOTAL</time><span>{stats.total} grabaciones disponibles</span></p></div></section>
      <section className="review-list-section"><button className="review-section-heading" onClick={() => setIsExpanded((current) => !current)} aria-expanded={isExpanded}><span><span className="review-section-icon"><Headphones size={18} /></span><span><strong>Revisar notas</strong><small>{isLoading ? 'Cargando notas...' : `${notes.length} notas esperan tu revisión`}</small></span></span><ChevronDown className={isExpanded ? 'rotate' : ''} size={21} /></button>{isExpanded && <div className="review-notes-list">{error && <p className="review-error-message">{error}</p>}{isLoading ? <div className="review-empty-state"><span>Cargando notas...</span></div> : notes.length ? notes.map((note) => <NoteRow key={note.id} note={note} playing={playingId === note.id} selectedClassification={selections[note.id]} onPlay={() => playAudio(note)} onSelect={(classification) => setSelections((current) => ({ ...current, [note.id]: classification }))} onConfirm={() => confirmClassification(note.id)} isSaving={savingId === note.id} />) : <div className="review-empty-state"><Check size={20} /><strong>Todo revisado</strong><span>Ya clasificaste todas las notas de esta sesión.</span></div>}</div>}</section>
    </div>
    {trainingStatus.status === 'training' && <aside className="review-training-tooltip" role="status" aria-live="polite"><div className="review-training-tooltip-header"><Sparkles size={16} /><strong>Entrenamiento en segundo plano</strong><span>{trainingStatus.progress}%</span></div><div className="review-training-progress"><i style={{ width: `${trainingStatus.progress}%` }} /></div><p>{trainingStatus.step}</p><small>No puedes iniciar otro entrenamiento hasta que este termine.</small><button className="review-training-cancel" onClick={cancelTraining} disabled={isCancellingTraining}>{isCancellingTraining ? 'Cancelando...' : 'Cancelar entrenamiento'}</button></aside>}
    {isTrainingOpen && createPortal(<div className="review-modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) closeTrainingModal(); }}><section className="review-training-modal" role="dialog" aria-modal="true" aria-labelledby="review-training-title"><button className="review-close-button" onClick={closeTrainingModal} aria-label="Cerrar"><X size={18} /></button><div className="review-modal-symbol"><Sparkles size={22} /></div><p className="review-eyebrow">NUEVA VERSIÓN</p><h2 id="review-training-title">Entrenar modelo</h2>
      {trainingResult ? (
        <div className="review-training-success">
          <p><strong>¡Modelo entrenado con éxito!</strong></p>
          <ul className="review-training-metrics">
            <li>Muestras usadas: {trainingResult.n_samples_human + trainingResult.n_samples_synthetic}</li>
            {trainingResult.val_accuracy !== null && <li>Precisión: {(trainingResult.val_accuracy * 100).toFixed(1)}%</li>}
            {trainingResult.val_auc !== null && <li>AUC: {trainingResult.val_auc.toFixed(3)}</li>}
          </ul>
          <button className="review-primary-button review-modal-submit" onClick={closeTrainingModal}>Cerrar</button>
        </div>
      ) : (
        <>
          <p className="review-modal-copy">Dale un nombre para reconocerlo fácilmente cuando esté listo.</p>
          {error && <p className="review-error-message" style={{marginBottom: '1rem'}}>{error}</p>}
          <form onSubmit={submitTraining}><label htmlFor="review-model-name">Nombre del modelo</label><div className="review-input-wrap"><input id="review-model-name" ref={inputRef} value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="Ej. Vocalis primavera" disabled={isTraining} /><span><CornerDownLeft size={13} />Intro</span></div><button className="review-primary-button review-modal-submit" type="submit" disabled={isTraining || !modelName.trim()}>{isTraining ? 'Entrenando...' : 'Comenzar entrenamiento'} <Sparkles size={16} /></button></form>
        </>
      )}
    </section></div>, document.body)}
  </div>;
}
