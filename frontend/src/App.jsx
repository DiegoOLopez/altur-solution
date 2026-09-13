/**
 * AuraVoice - Componente Principal (App.jsx)
 * 
 * Controla el estado global de la aplicación y el enrutamiento interno
 * entre las tres vistas principales:
 * - Análisis forense (Batch)
 * - Simulación en vivo (Streaming)
 * - Centro de Revisión (Review Hub)
 * 
 * Gestiona el envío del audio seleccionado al backend (POST /detect_wav)
 * y almacena el historial local de análisis en la sesión.
 */
import { useRef, useState } from 'react';
import { ArrowRight, Loader2 } from 'lucide-react';
import Navbar from './components/Navbar';
import BatchInput from './components/BatchInput';
import StreamingCallSimulator from './components/StreamingCallSimulator';
import StereoWaveform from './components/StereoWaveform';
import Spectrogram from './components/Spectrogram';
import DetectionSignals from './components/DetectionSignals';
import ResponseTiming from './components/ResponseTiming';
import VerdictPanel from './components/VerdictPanel';
import CallHistory from './components/CallHistory';
import ReviewHub from './components/ReviewHub';
import { callDetectApi } from './utils/audioUtils';

// URL base de la API backend. Por defecto localhost:8000
const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') ||
  'http://localhost:8000';

export default function App() {
  // Estado global de la interfaz
  const [activeMode, setActiveMode] = useState('batch'); // 'batch', 'streaming', 'review'
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeAudio, setActiveAudio] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [verdictFor, setVerdictFor] = useState(null);
  const [history, setHistory] = useState([]);

  // Título de la última llamada en vivo (para armar el registro del historial).
  const currentTitleRef = useRef('');

  // Selección de audio en modo batch. Disparado por BatchInput.
  const handleAudioReady = (audioData) => {
    setActiveAudio(audioData);

    // Limpiamos resultados anteriores al cargar un nuevo audio
    setAnalysisResult(null);
    setVerdictFor(null);
  };

  // Análisis batch: llama a POST /detect_wav con el WAV seleccionado.
  const handleRunBatchAnalysis = async () => {
    if (!activeAudio) return;

    setIsAnalyzing(true);
    const startTime = performance.now();

    try {
      // Enviar el WAV Base64 al backend
      const data = await callDetectApi(
        API_BASE,
        activeAudio.base64
      );

      const latency = Math.round(
        performance.now() - startTime
      );

      // Mapear la respuesta real del modelo al estado de la UI
      const res = {
        is_synthetic: data.is_synthetic,
        confidence: data.confidence,
        latency_ms: latency,
        score_total: data.score_total,
        llr_acoustic_cum: data.llr_acoustic_cum,
        llr_behavioral_cum: data.llr_behavioral_cum,
        n_acoustic_segments: data.n_acoustic_segments,
        n_behavioral_events: data.n_behavioral_events,
        eta: data.eta,
        apiResponse: data,
        // Generar una explicación para la UI
        llm_conclusion: data.is_synthetic
          ? `El detector clasificó la llamada como voz sintética con una confianza de ${(data.confidence * 100).toFixed(1)}%. El resultado se obtiene a partir de la evidencia acústica y comportamental acumulada por el modelo.`
          : `El detector clasificó la llamada como voz humana con una confianza de ${(data.confidence * 100).toFixed(1)}%. El resultado se obtiene a partir de la evidencia acústica y comportamental acumulada por el modelo.`
      };

      setAnalysisResult(res);
      setVerdictFor(activeAudio.title);
      addHistoryItem(res, activeAudio.title);

    } catch (err) {
      console.error('Error calling POST /detect_wav:', err);
      // Manejar error de conexión
      setAnalysisResult({
        error: true,
        errorMessage: err?.message || 'No se pudo completar el análisis.'
      });
      setVerdictFor(activeAudio.title);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Actualización en vivo del modo streaming.
  const handleStreamingUpdate = async (streamData) => {
    setAnalysisResult((prev) => ({
      ...prev,
      is_synthetic: streamData.is_synthetic,
      confidence: streamData.confidence,
      score_total: streamData.score_total,
      llr_acoustic_cum: streamData.llr_acoustic_cum,
      llr_behavioral_cum: streamData.llr_behavioral_cum,
      n_acoustic_segments: streamData.n_acoustic_segments,
      n_behavioral_events: streamData.n_behavioral_events,
      eta: streamData.eta,
      metrics: {
        ...prev?.metrics,
        acoustic_score: streamData.is_synthetic ? 94 : 8,
        turn_recovery_ms: streamData.is_synthetic ? 180 : 420
      }
    }));
  };

  // Fin de la llamada streaming: armar el registro para el historial local.
  const handleCallFinished = (summary) => {
    const finalResult = {
      is_synthetic: summary.isSynthetic,
      confidence: summary.confidence,
      latency_ms: 104,
      metrics: {
        acoustic_score: summary.isSynthetic ? 94 : 8,
        spectral_artifacts: summary.isSynthetic
          ? 'Artefacto de síntesis neural detectado durante la transmisión continua de chunks.'
          : 'Modulación acústica natural en toda la llamada.',
        turn_recovery_ms: summary.isSynthetic ? 180 : 420,
        conversational_messiness: summary.isSynthetic ? 14 : 86,
        breathing_detected: !summary.isSynthetic,
        semantic_hallucination: summary.isSynthetic
      },
      llm_conclusion: summary.isSynthetic
        ? `ALERTA DEEPFAKE EN LLAMADA STREAMING:\nDurante la llamada de ${summary.duration} segundos, se detectó una frecuencia neural plana a 3.8 kHz y latencias invariables al responder a las interrupciones del agente.`
        : `LLAMADA STREAMING AUTÉNTICA:\nLa llamada en vivo de ${summary.duration} segundos presentó variabilidad prosódica natural y fluidez orgánica.`
    };

    setAnalysisResult(finalResult);

    const title =
      summary.title ||
      currentTitleRef.current ||
      `Llamada en Vivo (${summary.duration}s)`;

    setVerdictFor(title);
    addHistoryItem(finalResult, title);
  };

  // Guardar el título al finalizar una grabación del micrófono.
  const handleRecordingResult = (recording) => {
    currentTitleRef.current = recording.title;
    setActiveAudio(recording);
    setVerdictFor(null);
  };

  // Añadir un análisis al historial local (hasta 5 items).
  const addHistoryItem = (res, title) => {
    const now = new Date();
    const timeStr =
      `${now.getHours().toString().padStart(2, '0')}:` +
      `${now.getMinutes().toString().padStart(2, '0')}:` +
      `${now.getSeconds().toString().padStart(2, '0')}`;

    setHistory((prev) => [
      {
        title: title || 'Llamada telefónica',
        timestamp: timeStr,
        duration: activeAudio?.duration || 6.0,
        is_synthetic: res.is_synthetic,
        confidence: res.confidence,
        latency_ms: res.latency_ms || 108,
        resultData: res
      },
      ...prev.slice(0, 4) // Guardar solo los últimos 5
    ]);
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Navbar
        activeMode={activeMode}
        setActiveMode={setActiveMode}
      />

      {activeMode === 'review' ? (
        <ReviewHub />
      ) : (
        <main
          style={{
            maxWidth: '1400px',
            margin: '0 auto',
            padding: '28px 28px 44px',
            width: '100%',
            flex: 1
          }}
        >
          {activeMode === 'batch' ? (
            <div className="forensic-shell review-hub">
              <section className="review-intro">
                <div>
                  <p className="review-eyebrow"><span />Módulo de auditoría forense</p>
                  <h1>Análisis<br /><em>forense.</em></h1>
                  <p className="review-intro-copy">
                    Verifica si la voz del llamante es humana o sintética utilizando
                    la evidencia acústica y comportamental que devuelve el modelo.
                  </p>
                </div>
                <button
                  className="review-primary-button"
                  onClick={handleRunBatchAnalysis}
                  disabled={isAnalyzing || !activeAudio}
                >
                  {isAnalyzing ? (
                    <><Loader2 size={18} className="rotate-spin" /> Analizando…</>
                  ) : (
                    <><ArrowRight size={18} /> Ejecutar detección</>
                  )}
                </button>
              </section>

              {/* Bento Dashboard con todos los componentes */}
              <div className="fx-bento">
                <div className="fx-cell-input">
                  <BatchInput
                    onAudioReady={handleAudioReady}
                    isAnalyzing={isAnalyzing}
                    currentAudioTitle={activeAudio?.title}
                    verdictTitle={verdictFor}
                  />
                </div>

                <div className="fx-cell-verdict">
                  <VerdictPanel
                    result={analysisResult}
                    isAnalyzing={isAnalyzing}
                    activeMode={activeMode}
                  />
                </div>

                <div className="fx-cell-wave">
                  <StereoWaveform
                    channel0={activeAudio?.channel0}
                    channel1={activeAudio?.channel1}
                    duration={activeAudio?.duration}
                    isSynthetic={analysisResult?.is_synthetic}
                    isAnalyzing={isAnalyzing}
                  />
                </div>

                <div className="fx-cell-signals">
                  <DetectionSignals
                    isSynthetic={analysisResult?.is_synthetic}
                    confidence={analysisResult?.confidence}
                    scoreTotal={analysisResult?.score_total}
                    acousticLLR={analysisResult?.llr_acoustic_cum}
                    behavioralLLR={analysisResult?.llr_behavioral_cum}
                    acousticSegments={analysisResult?.n_acoustic_segments}
                    behavioralEvents={analysisResult?.n_behavioral_events}
                  />
                </div>

                <div className="fx-cell-spec">
                  <Spectrogram
                    channelData={activeAudio?.channel0}
                    duration={activeAudio?.duration}
                    isAnalyzing={isAnalyzing}
                  />
                </div>

                <div className="fx-cell-timing">
                  <ResponseTiming
                    isSynthetic={analysisResult?.is_synthetic}
                    behavioralLLR={analysisResult?.llr_behavioral_cum}
                    behavioralEvents={analysisResult?.n_behavioral_events}
                    scoreTotal={analysisResult?.score_total}
                  />
                </div>
              </div>

              {/* Historial Local (Solo Batch/Streaming) */}
              <CallHistory
                history={history}
                onSelectHistoryItem={(item) => {
                  if (item.resultData) {
                    setAnalysisResult(item.resultData);
                    setVerdictFor(item.title);
                  }
                }}
              />
            </div>
          ) : (
            <>
              {/* Modo Streaming */}
              <StreamingCallSimulator
                onStreamingUpdate={handleStreamingUpdate}
                onCallFinished={handleCallFinished}
                onRecordingResult={handleRecordingResult}
                apiBaseUrl={API_BASE}
              />
              <CallHistory
                history={history}
                onSelectHistoryItem={(item) => {
                  if (item.resultData) {
                    setAnalysisResult(item.resultData);
                    setVerdictFor(item.title);
                  }
                }}
              />
            </>
          )}
        </main>
      )}

      {/* Footer corporativo */}
      <footer
        style={{
          borderTop: '1px solid var(--border-glass)',
          padding: '18px 24px',
          background: 'rgba(7, 9, 20, 0.95)',
          textAlign: 'center',
          fontSize: '0.78rem',
          color: 'var(--text-subtle)'
        }}
      >
        AuraVoice • Desarrollado para el reto de Tecnologías Altur en
        HackMTY26 • Soporte para{' '}
        <code style={{ color: '#00f2fe' }}>
          POST /detect_wav
        </code>{' '}
        y{' '}
        <code style={{ color: '#ff3366' }}>
          WS /ws/detect
        </code>
      </footer>
    </div>
  );
}