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

  const connectWebSocket = () => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        return Promise.resolve(wsRef.current);
      }

      return Promise.reject(new Error("La conexión ya está iniciándose."));
    }

    const wsUrl = `${apiBaseUrl.replace(
      /^http/,
      "ws"
    )}/ws/detect`;

    const ws = new WebSocket(wsUrl);

    wsRef.current = ws;

    ws.binaryType = "arraybuffer";

    return new Promise((resolve, reject) => {
      ws.onopen = () => {
      console.log("WebSocket conectado");

      setIsConnected(true);
        resolve(ws);
      };

    ws.onmessage = (message) => {
      if (typeof message.data !== "string") {
        return;
      }

      try {
        const data = JSON.parse(message.data);

        console.log("Backend →", data);

        switch (data.event) {
          case "ready":
            console.log("Audio listo:", data);
            break;

          case "chunk_processed":
            break;

          case "detection":
            setDetection(data.result);
            break;

          case "error":
            console.error(
              "Backend error:",
              data.message
            );
            break;

          default:
            console.log(
              "Evento no manejado:",
              data
            );
        }
      } catch (error) {
        console.error(
          "Respuesta inválida:",
          error
        );
      }
    };

    ws.onerror = (error) => {
      console.error(
        "WebSocket error:",
        error
      );
      wsRef.current = null;
      reject(new Error("No se pudo conectar con el backend."));
    };

    ws.onclose = () => {
      console.log(
        "WebSocket desconectado"
      );

      wsRef.current = null;

      setIsConnected(false);
      setIsRecording(false);
    };
    });
  };

  const startRecording = async () => {
    if (isRecording || isStarting) {
      return;
    }

    setIsStarting(true);

    try {
      const ws = await connectWebSocket();
      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: false,
              autoGainControl: false,
            },
          }
        );

      streamRef.current = stream;

      const audioContext =
        new AudioContext({
          sampleRate: 16000,
        });

      audioContextRef.current =
        audioContext;

      await audioContext.resume();

      const realSampleRate =
        audioContext.sampleRate;

      setSampleRate(
        realSampleRate
      );

      /*
       * Primero enviamos al backend
       * la configuración del audio.
       */
      ws.send(
        JSON.stringify({
          sample_rate:
            realSampleRate,
          channels: 1,
          sample_width: 2,
        })
      );

      await audioContext.audioWorklet.addModule(
        "/audio-processor.js"
      );

      const source =
        audioContext.createMediaStreamSource(
          stream
        );

      const worklet =
        new AudioWorkletNode(
          audioContext,
          "pcm-processor"
        );

      workletRef.current = worklet;

      /*
       * Cada mensaje contiene
       * PCM Int16.
       */
      worklet.port.onmessage = (
        event
      ) => {
        const pcmBuffer =
          event.data;

        if (
          ws.readyState !==
          WebSocket.OPEN
        ) {
          return;
        }

        ws.send(pcmBuffer);

        setChunksSent(
          (prev) => prev + 1
        );
      };

      source.connect(worklet);

      setIsRecording(true);

      console.log(
        "Micrófono iniciado:",
        realSampleRate,
        "Hz"
      );
    } catch (error) {
      console.error(
        "No se pudo iniciar la grabación:",
        error
      );
      stop();
    } finally {
      setIsStarting(false);
    }
  };

  const stop = () => {
    if (workletRef.current) {
      workletRef.current.disconnect();
      workletRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current
        .close()
        .catch(() => {});

      audioContextRef.current = null;
    }

    if (
      wsRef.current &&
      wsRef.current.readyState ===
        WebSocket.OPEN
    ) {
      wsRef.current.close();
    }

    wsRef.current = null;

    setIsRecording(false);
    setIsConnected(false);
    setSampleRate(null);
    setChunksSent(0);
    setDetection(null);
  };

  useEffect(() => {
    return () => {
      stop();
    };
  }, []);

  const confidence = detection?.confidence ?? null;
  const isSynthetic = detection?.is_synthetic ?? null;

  return (
    <section className="streaming-recorder">
      <header className="streaming-recorder__header">
        <div>
          <span className="streaming-recorder__eyebrow">STREAMING CALL SIMULATOR</span>
          <h2>Centro de monitoreo</h2>
        </div>
        <span className={`streaming-recorder__status ${isRecording ? "is-live" : ""}`}>
          <span className="streaming-recorder__status-dot" />
          {isRecording ? "En vivo" : isStarting ? "Preparando" : "En espera"}
        </span>
      </header>

      <div className="streaming-recorder__bento">
        <div className={`streaming-recorder__tile streaming-recorder__tile--control ${isRecording ? "is-live" : ""}`}>
          <div className="streaming-recorder__tile-label"><Mic size={15} /> CAPTURA DE AUDIO</div>
          <div className="streaming-recorder__control-copy">
            <strong>{isRecording ? "Transmisión activa" : "Micrófono listo"}</strong>
            <span>{isRecording ? "Enviando señal al detector" : "Inicia una llamada para analizarla"}</span>
          </div>
          <button
            className={`streaming-recorder__button ${isRecording ? "is-stop" : ""}`}
            onClick={isRecording ? stop : startRecording}
            disabled={isStarting}
          >
            {isRecording ? <CircleStop size={17} /> : <Mic size={17} />}
            {isStarting ? "Conectando..." : isRecording ? "Detener grabación" : "Empezar a grabar"}
          </button>
        </div>

        <div className="streaming-recorder__tile streaming-recorder__tile--server">
          <div className="streaming-recorder__tile-label"><Server size={15} /> BACKEND</div>
          <div className="streaming-recorder__metric-icon"><Wifi size={19} /></div>
          <strong>{isConnected ? "Conectado" : "Desconectado"}</strong>
          <span>WebSocket /detect</span>
        </div>

        <div className="streaming-recorder__tile streaming-recorder__tile--signal">
          <div className="streaming-recorder__tile-label"><Radio size={15} /> SEÑAL ENTRANTE</div>
          <div className="streaming-recorder__wave" aria-hidden="true">
            {[12, 24, 38, 20, 50, 29, 42, 17, 31, 46, 23, 36, 15].map((height, index) => (
              <span key={index} style={{ height: `${height}px` }} />
            ))}
          </div>
          <div className="streaming-recorder__signal-caption">
            <strong>{sampleRate ?? "-"} Hz</strong>
            <span>{chunksSent} chunks enviados</span>
          </div>
        </div>

        <div className="streaming-recorder__tile streaming-recorder__tile--result">
          <div className="streaming-recorder__result-heading">
            <div className="streaming-recorder__tile-label"><Activity size={15} /> DETECCIÓN</div>
            <span>{detection ? "Actualizado" : "Esperando señal"}</span>
          </div>
          {!detection ? (
            <p>La primera inferencia aparecerá durante la grabación.</p>
          ) : (
            <>
              <div className="streaming-recorder__verdict">
                {isSynthetic ? "Voz sintética" : "Voz humana"}
              </div>
              <div className="streaming-recorder__details">
                <span><strong>Confianza</strong> {confidence !== null ? `${(confidence * 100).toFixed(2)}%` : "-"}</span>
                <span><strong>Tiempo</strong> {detection.t}s</span>
                <span><strong>Segmentos</strong> {detection.n_acoustic_segments ?? detection.segments_analyzed ?? "-"}</span>
                <span><strong>Score</strong> {detection.score_total ?? "-"}</span>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}