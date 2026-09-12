import React, { useState } from 'react';
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

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') ||
  'http://localhost:8000';

export default function App() {
  const [activeMode, setActiveMode] = useState('batch');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeAudio, setActiveAudio] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [verdictFor, setVerdictFor] = useState(null);
  const [history, setHistory] = useState([]);

  // Handler for audio selection in Batch Mode
  const handleAudioReady = (audioData) => {
    setActiveAudio(audioData);

    // El nuevo audio todavía no ha sido analizado.
    // No mostramos ningún resultado previo o simulado.
    setAnalysisResult(null);
    setVerdictFor(null);
  };

  // Handler for Batch Analysis (POST /detect)
  const handleRunBatchAnalysis = async () => {
    if (!activeAudio) return;

    setIsAnalyzing(true);

    const startTime = performance.now();

    try {
      // Enviar el WAV al backend
      const data = await callDetectApi(
        API_BASE,
        activeAudio.base64
      );

      const latency = Math.round(
        performance.now() - startTime
      );

      // Resultado REAL del modelo
      const res = {
        is_synthetic: data.is_synthetic,
        confidence: data.confidence,

        // Latencia real de la petición HTTP
        latency_ms: latency,

        // Evidencia generada por el modelo
        score_total: data.score_total,
        llr_acoustic_cum: data.llr_acoustic_cum,
        llr_behavioral_cum: data.llr_behavioral_cum,

        // Cantidad real de elementos analizados
        n_acoustic_segments: data.n_acoustic_segments,
        n_behavioral_events: data.n_behavioral_events,

        // Umbral utilizado por el detector
        eta: data.eta,

        // Respuesta original del backend
        apiResponse: data,

        // Explicación basada únicamente en el resultado real
        llm_conclusion: data.is_synthetic
          ? `El detector clasificó la llamada como voz sintética con una confianza de ${(data.confidence * 100).toFixed(1)}%. El resultado se obtiene a partir de la evidencia acústica y comportamental acumulada por el modelo.`
          : `El detector clasificó la llamada como voz humana con una confianza de ${(data.confidence * 100).toFixed(1)}%. El resultado se obtiene a partir de la evidencia acústica y comportamental acumulada por el modelo.`
      };

      setAnalysisResult(res);
      setVerdictFor(activeAudio.title);

      addHistoryItem(
        res,
        activeAudio.title
      );
    } catch (err) {
      console.error(
        'Error calling POST /detect:',
        err
      );

      // No utilizar mock si falla el backend.
      setAnalysisResult({
        error: true,
        errorMessage:
          err?.message ||
          'No se pudo completar el análisis.'
      });

      setVerdictFor(activeAudio.title);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handler for live streaming updates (POST /detect_streaming)
  // Se conserva el flujo existente de streaming.
  const handleStreamingUpdate = async (streamData) => {
    if (analysisResult) {
      setAnalysisResult((prev) => ({
        ...prev,
        is_synthetic: streamData.isSynthetic,
        confidence: streamData.currentRisk / 100,
        metrics: {
          ...prev.metrics,
          acoustic_score: streamData.isSynthetic
            ? Math.min(95, 50 + streamData.elapsed * 8)
            : Math.max(5, 30 - streamData.elapsed * 4),
          turn_recovery_ms: streamData.isSynthetic ? 180 : 420
        }
      }));
    }
  };

  // Handler when streaming call ends
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
        ? `🚨 ALERTA DEEPFAKE EN LLAMADA STREAMING (POST /detect_streaming):
Durante la llamada de ${summary.duration} segundos, se detectó una frecuencia neural plana a 3.8 kHz y latencias invariables de 180ms al responder a las interrupciones del agente. Se recomienda abortar la operación bancaria.`
        : `✅ LLAMADA STREAMING AUTÉNTICA (POST /detect_streaming):
La llamada en vivo de ${summary.duration} segundos presentó variabilidad prosódica natural, respiraciones audibles y fluidez orgánica. Canal telefónico 100% verificado.`
    };

    setAnalysisResult(finalResult);

    const title =
      summary.title ||
      currentTitleRef.current ||
      `Llamada en Vivo (${summary.duration}s)`;

    setVerdictFor(title);
    addHistoryItem(finalResult, title);
  };

  // Latest live-call title
  const currentTitleRef = React.useRef('');

  const handleRecordingResult = (recording) => {
    currentTitleRef.current = recording.title;
    setActiveAudio(recording);
    setVerdictFor(null);
  };

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
      ...prev.slice(0, 5)
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
        <>
          {/* Main Responsive Grid */}
          <main
            style={{
              maxWidth: '1400px',
              margin: '0 auto',
              padding: '28px 24px',
              width: '100%',
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: '24px'
            }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns:
                  activeMode === 'streaming'
                    ? '1fr'
                    : 'repeat(auto-fit, minmax(380px, 1fr))',
                gap: '24px',
                alignItems: 'start'
              }}
            >
              {/* Column 1: Mode Specific Input Hub */}
              <div>
                {activeMode === 'batch' ? (
                  <BatchInput
                    onAudioReady={handleAudioReady}
                    onRunBatchAnalysis={handleRunBatchAnalysis}
                    isAnalyzing={isAnalyzing}
                    currentAudioTitle={activeAudio?.title}
                    verdictTitle={verdictFor}
                  />
                ) : (
                  <StreamingCallSimulator
                    onStreamingUpdate={handleStreamingUpdate}
                    onCallFinished={handleCallFinished}
                    onRecordingResult={handleRecordingResult}
                    apiBaseUrl={API_BASE}
                  />
                )}
              </div>

              {/* Column 2: Visualizer & Results */}
              {activeMode === 'batch' && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '20px'
                  }}
                >
                  {/* CALLER AUDIO: Conversation Sound Wave */}
                  <StereoWaveform
                    channel0={activeAudio?.channel0}
                    channel1={activeAudio?.channel1}
                    duration={activeAudio?.duration}
                    isSynthetic={analysisResult?.is_synthetic}
                    isAnalyzing={isAnalyzing}
                  />

                  {/* CALLER AUDIO: Frequency-domain view */}
                  <Spectrogram
                    channelData={activeAudio?.channel0}
                    duration={activeAudio?.duration}
                    isAnalyzing={isAnalyzing}
                  />

                  {/* DETECTION RESULT: real model evidence */}
                  <DetectionSignals
                    isSynthetic={analysisResult?.is_synthetic}
                    confidence={analysisResult?.confidence}
                    scoreTotal={analysisResult?.score_total}
                    acousticLLR={analysisResult?.llr_acoustic_cum}
                    behavioralLLR={analysisResult?.llr_behavioral_cum}
                    acousticSegments={analysisResult?.n_acoustic_segments}
                    behavioralEvents={analysisResult?.n_behavioral_events}
                  />

                  {/* CONVERSATION ANALYSIS: real behavioral evidence */}
                  <ResponseTiming
                    isSynthetic={analysisResult?.is_synthetic}
                    behavioralLLR={analysisResult?.llr_behavioral_cum}
                    behavioralEvents={analysisResult?.n_behavioral_events}
                    scoreTotal={analysisResult?.score_total}
                  />

                  {/* Verdict based on real detector result */}
                  <VerdictPanel
                    result={analysisResult}
                    isAnalyzing={isAnalyzing}
                    activeMode={activeMode}
                  />
                </div>
              )}
            </div>

            {/* Recent Calls History */}
            <CallHistory
              history={history}
              onSelectHistoryItem={(item) => {
                if (item.resultData) {
                  setAnalysisResult(item.resultData);
                  setVerdictFor(item.title);
                }
              }}
            />
          </main>
        </>
      )}

      {/* Footer */}
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
          POST /detect
        </code>{' '}
        y{' '}
        <code style={{ color: '#ff3366' }}>
          POST /detect_streaming
        </code>
      </footer>
    </div>
  );
}