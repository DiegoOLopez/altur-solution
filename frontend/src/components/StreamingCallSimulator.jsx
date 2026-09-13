import React, { useEffect, useRef, useState } from "react";
import { Activity, CircleStop, Mic, Radio, Server, Wifi } from "lucide-react";

export default function StreamingCallSimulator({
  apiBaseUrl = "http://localhost:8000",
}) {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [sampleRate, setSampleRate] = useState(null);
  const [chunksSent, setChunksSent] = useState(0);

  const [detection, setDetection] = useState(null);

  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const workletRef = useRef(null);

  // Abre el WebSocket /ws/detect del backend y devuelve la conexión.
  const connectWebSocket = () => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        return Promise.resolve(wsRef.current);
      }

      return Promise.reject(new Error("La conexión ya está iniciándose."));
    }

    const wsUrl = `${apiBaseUrl.replace(/^http/, "ws")}/ws/detect`;

    const ws = new WebSocket(wsUrl);

    wsRef.current = ws;

    ws.binaryType = "arraybuffer";

    return new Promise((resolve, reject) => {
      ws.onopen = () => {
        setIsConnected(true);
        resolve(ws);
      };

      ws.onmessage = (message) => {
        if (typeof message.data !== "string") {
          return;
        }

        try {
          const data = JSON.parse(message.data);

          switch (data.event) {
            case "ready":
              break;

            case "chunk_processed":
              break;

            case "detection":
              setDetection(data.result);
              break;

            case "error":
              console.error("Backend error:", data.message);
              break;

            default:
              break;
          }
        } catch (error) {
          console.error("Respuesta inválida:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        wsRef.current = null;
        reject(new Error("No se pudo conectar con el backend."));
      };

      ws.onclose = () => {
        wsRef.current = null;

        setIsConnected(false);
        setIsRecording(false);
      };
    });
  };

  // Inicia la captura del micrófono y la transmisión en vivo.
  const startRecording = async () => {
    if (isRecording || isStarting) {
      return;
    }

    setIsStarting(true);

    try {
      const ws = await connectWebSocket();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      streamRef.current = stream;

      const audioContext = new AudioContext({
        sampleRate: 16000,
      });

      audioContextRef.current = audioContext;

      await audioContext.resume();

      const realSampleRate = audioContext.sampleRate;

      setSampleRate(realSampleRate);

      /*
       * Primero enviamos al backend
       * la configuración del audio.
       */
      ws.send(
        JSON.stringify({
          sample_rate: realSampleRate,
          channels: 1,
          sample_width: 2,
        })
      );

      await audioContext.audioWorklet.addModule(
        "/audio-processor.js"
      );

      const source = audioContext.createMediaStreamSource(stream);

      const worklet = new AudioWorkletNode(
        audioContext,
        "pcm-processor"
      );

      workletRef.current = worklet;

      /*
       * Cada mensaje contiene
       * PCM Int16.
       */
      worklet.port.onmessage = (event) => {
        const pcmBuffer = event.data;

        if (ws.readyState !== WebSocket.OPEN) {
          return;
        }

        ws.send(pcmBuffer);

        setChunksSent((prev) => prev + 1);
      };

      source.connect(worklet);

      setIsRecording(true);
    } catch (error) {
      console.error("No se pudo iniciar la grabación:", error);
      stop();
    } finally {
      setIsStarting(false);
    }
  };

  // Detiene la captura, la conexión y el contexto de audio.
  const stop = () => {
    if (workletRef.current) {
      workletRef.current.disconnect();
      workletRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});

      audioContextRef.current = null;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    wsRef.current = null;

    setIsRecording(false);
    setIsConnected(false);
    setSampleRate(null);
    setChunksSent(0);
    setDetection(null);
  };

  // Al desmontar el componente se limpian todos los recursos.
  useEffect(() => {
    return () => {
      stop();
    };
  }, []);

  const confidence = detection?.confidence ?? null;
  const isSynthetic = detection?.is_synthetic ?? null;

  return (
    <div className="review-hub">
      <section className="review-intro">
        <div>
          <p className="review-eyebrow"><span />Simulador de Llamada en Vivo</p>
          <h1>Centro de<br /><em>monitoreo.</em></h1>
          <p className="review-intro-copy">Inicia una llamada simulada para probar el detector de fraude en tiempo real.</p>
        </div>
        <button 
          className="review-primary-button" 
          onClick={isRecording ? stop : startRecording}
          disabled={isStarting}
        >
          {isRecording ? <CircleStop size={18} /> : <Mic size={18} />}
          {isStarting ? "Conectando..." : isRecording ? "Detener grabación" : "Empezar a grabar"}
        </button>
      </section>

      <div className="sim-bento-grid">
        <section className="sim-card" style={{ gridArea: 'capture' }}>
          <div className="sim-card-header"><Mic size={16} /> Captura de audio</div>
          <div className="sim-card-content" style={{ textAlign: 'center', justifyContent: 'center' }}>
            <div className={`sim-status ${isRecording ? 'connected' : 'disconnected'}`}>
              <Mic size={24} className={isRecording ? 'animate-pulse' : ''} />
            </div>
            <p className="sim-value-large">{isRecording ? "Transmisión activa" : "Micrófono listo"}</p>
            <p className="sim-value-label">{isRecording ? "Enviando señal al detector" : "Inicia una llamada para analizarla"}</p>
          </div>
        </section>

        <section className="sim-card" style={{ gridArea: 'backend' }}>
          <div className="sim-card-header"><Server size={16} /> Servidor</div>
          <div className="sim-card-content" style={{ textAlign: 'center', justifyContent: 'center' }}>
            <div className={`sim-status ${isConnected ? 'connected' : 'disconnected'}`}>
              <Wifi size={24} />
            </div>
            <p className="sim-value-large">{isConnected ? "Conectado" : "Desconectado"}</p>
            <p className="sim-value-label">WebSocket /detect</p>
          </div>
        </section>

        <section className="sim-card" style={{ gridArea: 'signal' }}>
          <div className="sim-card-header"><Radio size={16} /> Señal Entrante</div>
          <div className="sim-card-content" style={{ justifyContent: 'center' }}>
            <div className="sim-wave-bars" aria-hidden="true">
              {[12, 24, 38, 20, 48, 29, 42, 17, 31, 46].map((height, i) => (
                <span 
                  key={i} 
                  className={isRecording ? 'active' : ''}
                  style={{ height: `${height}px`, animationDelay: `${i * 0.1}s` }} 
                />
              ))}
            </div>
            <p className="sim-value-large">{sampleRate ?? "-"} Hz</p>
            <p className="sim-value-label">{chunksSent} chunks enviados</p>
          </div>
        </section>

        <section className="sim-card" style={{ gridArea: 'detection' }}>
          <div className="sim-card-header" style={{ justifyContent: 'space-between', marginBottom: '16px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Activity size={16} /> Detección</span>
            <span className={`sim-detection-badge ${detection ? 'updated' : 'waiting'}`}>
              {detection ? "Actualizado" : "Esperando"}
            </span>
          </div>
          
          {!detection ? (
            <div className="sim-card-content" style={{ justifyContent: 'center', alignItems: 'center' }}>
               <p className="sim-value-label">La primera inferencia aparecerá durante la grabación.</p>
            </div>
          ) : (
            <div className="sim-card-content">
              <div className={`sim-verdict-badge ${isSynthetic ? 'synthetic' : 'human'}`}>
                {isSynthetic ? "Voz sintética" : "Voz humana"}
              </div>
              
              <div className="sim-metrics-grid">
                <div className="sim-metric">
                  <span className="sim-metric-label">Confianza</span>
                  <span className="sim-metric-value">{confidence !== null ? `${(confidence * 100).toFixed(2)}%` : "-"}</span>
                </div>
                <div className="sim-metric">
                  <span className="sim-metric-label">Tiempo</span>
                  <span className="sim-metric-value">{detection.t}s</span>
                </div>
                <div className="sim-metric">
                  <span className="sim-metric-label">Segmentos</span>
                  <span className="sim-metric-value">{detection.n_acoustic_segments ?? detection.segments_analyzed ?? "-"}</span>
                </div>
                <div className="sim-metric">
                  <span className="sim-metric-label">Puntuación</span>
                  <span className="sim-metric-value">{detection.score_total ?? "-"}</span>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}