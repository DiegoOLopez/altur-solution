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
    <section className="streaming-call-simulator bg-white text-gray-900 font-sans py-20 px-4 md:px-8 max-w-6xl mx-auto">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 gap-6">
        <div>
          <span className="text-gray-500 text-sm font-semibold tracking-wider uppercase mb-2 block">STREAMING CALL SIMULATOR</span>
          <h2 className="font-serif text-4xl md:text-5xl font-bold">Centro de monitoreo</h2>
        </div>
        <span className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${isRecording ? "bg-green-50 text-green-700 ring-1 ring-green-600/20" : isStarting ? "bg-amber-50 text-amber-700 ring-1 ring-amber-600/20" : "bg-gray-100 text-gray-600"}`}>
          <span className={`w-2 h-2 rounded-full ${isRecording ? "bg-green-500 animate-pulse" : isStarting ? "bg-amber-500 animate-pulse" : "bg-gray-400"}`} />
          {isRecording ? "En vivo" : isStarting ? "Preparando" : "En espera"}
        </span>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <div className={`bg-gray-50 rounded-2xl p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-col justify-between ${isRecording ? "ring-2 ring-green-500/20" : ""}`}>
          <div>
            <div className="flex items-center gap-2 text-gray-500 text-sm font-medium mb-4 uppercase tracking-wide"><Mic size={16} /> Captura de audio</div>
            <div className="mb-8 flex flex-col gap-1">
              <strong className="text-gray-900 text-lg">{isRecording ? "Transmisión activa" : "Micrófono listo"}</strong>
              <span className="text-gray-500 text-sm">{isRecording ? "Enviando señal al detector" : "Inicia una llamada para analizarla"}</span>
            </div>
          </div>
          <button
            className={`w-full flex items-center justify-center gap-2 rounded-full px-6 py-3 font-medium transition-all ${isRecording ? "bg-gray-200 text-gray-900 hover:bg-gray-300" : "bg-black text-white hover:bg-gray-800"}`}
            onClick={isRecording ? stop : startRecording}
            disabled={isStarting}
          >
            {isRecording ? <CircleStop size={18} /> : <Mic size={18} />}
            {isStarting ? "Conectando..." : isRecording ? "Detener grabación" : "Empezar a grabar"}
          </button>
        </div>

        <div className="bg-gray-50 rounded-2xl p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-col">
          <div className="flex items-center gap-2 text-gray-500 text-sm font-medium mb-4 uppercase tracking-wide"><Server size={16} /> Backend</div>
          <div className="flex-1 flex flex-col justify-center items-center text-center py-4">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-4 ${isConnected ? "bg-green-100 text-green-600" : "bg-gray-200 text-gray-400"}`}>
              <Wifi size={24} />
            </div>
            <strong className="text-gray-900 text-lg mb-1">{isConnected ? "Conectado" : "Desconectado"}</strong>
            <span className="text-gray-500 text-sm">WebSocket /detect</span>
          </div>
        </div>

        <div className="bg-gray-50 rounded-2xl p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-col">
          <div className="flex items-center gap-2 text-gray-500 text-sm font-medium mb-4 uppercase tracking-wide"><Radio size={16} /> Señal Entrante</div>
          <div className="flex-1 flex flex-col justify-center py-4">
            <div className="flex items-end justify-center gap-1.5 h-16 mb-6 opacity-60" aria-hidden="true">
              {[12, 24, 38, 20, 50, 29, 42, 17, 31, 46, 23, 36, 15].map((height, index) => (
                <span key={index} className={`w-1.5 rounded-t-sm ${isRecording ? "bg-green-500 animate-pulse" : "bg-gray-300"}`} style={{ height: `${height}px`, animationDelay: `${index * 0.1}s` }} />
              ))}
            </div>
            <div className="flex flex-col text-center">
              <strong className="text-gray-900 text-lg mb-1">{sampleRate ?? "-"} Hz</strong>
              <span className="text-gray-500 text-sm">{chunksSent} chunks enviados</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded-2xl p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-gray-500 text-sm font-medium uppercase tracking-wide"><Activity size={16} /> Detección</div>
            <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${detection ? "bg-green-100 text-green-700" : "bg-gray-200 text-gray-600"}`}>
              {detection ? "Actualizado" : "Esperando"}
            </span>
          </div>

          {!detection ? (
            <div className="flex-1 flex items-center justify-center text-center py-4">
              <p className="text-gray-500 text-sm leading-relaxed">La primera inferencia aparecerá durante la grabación.</p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col justify-center">
              <div className={`text-center py-3 px-4 rounded-xl mb-6 font-medium text-lg ${isSynthetic ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"}`}>
                {isSynthetic ? "Voz sintética" : "Voz humana"}
              </div>
              <div className="grid grid-cols-2 gap-y-4 gap-x-2 text-sm">
                <div className="flex flex-col">
                  <span className="text-gray-500 mb-0.5">Confianza</span>
                  <strong className="text-gray-900">{confidence !== null ? `${(confidence * 100).toFixed(2)}%` : "-"}</strong>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 mb-0.5">Tiempo</span>
                  <strong className="text-gray-900">{detection.t}s</strong>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 mb-0.5">Segmentos</span>
                  <strong className="text-gray-900">{detection.n_acoustic_segments ?? detection.segments_analyzed ?? "-"}</strong>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 mb-0.5">Score</span>
                  <strong className="text-gray-900">{detection.score_total ?? "-"}</strong>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}