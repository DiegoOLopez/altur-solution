"""
Detector de voz sintética de AuraVoice.

Es el corazón del sistema: carga el modelo entrenado ``ai_models/altur_detector_model.joblib``
una sola vez al arrancar el servicio y expone dos formas de análisis:

- ``detect_offline`` (lote): analiza una llamada completa. Se usa en
  ``POST /detect`` y ``POST /detect_wav``, recibiendo el WAV estéreo
  original (Canal 0 = llamante, Canal 1 = agente). El modelo se encarga
  internamente de cargar el WAV, resamplear a 16 kHz, separar los canales,
  extraer las características acústicas con Wav2Vec2, calcular el score
  y devolver el veredicto final.

- ``process_audio`` (streaming): analiza chunks en tiempo real. Se usa en
  el WebSocket ``/ws/detect`` y recibe PCM mono de 16 kHz. Cada chunk
  alimenta una sesión de streaming que acumula evidencia y devuelve un
  snapshot parcial cada ~3 segundos.

Pipeline de inferencia:
  1. Cargar audio (soundfile).
  2. Extraer canal 0 si es estéreo.
  3. Resamplear a 16 kHz (torchaudio).
  4. Truncar a 3 segundos (48,000 muestras).
  5. Tokenizar con Wav2Vec2Processor.
  6. Forward pass por Wav2Vec2Model (capa oculta 4).
  7. Promedio temporal → vector de 768 features.
  8. Clasificar con LogisticRegression (joblib).
"""
from pathlib import Path
import io
import sys

import joblib
import numpy as np
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# ------------------------------------------------------------------
# Ubicación del detector externo (paquete ``detector/``)
# ------------------------------------------------------------------
# El modelo base se encuentra en el directorio ``detector/`` de la raíz
# del repositorio. Se agrega al sys.path para permitir importaciones
# directas sin necesidad de instalarlo como paquete.
DETECTOR_ROOT = Path(__file__).resolve().parents[4] / "detector"

if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))


# ------------------------------------------------------------------
# Ruta al modelo serializado (.joblib)
# ------------------------------------------------------------------
# El clasificador entrenado (LogisticRegression serializado con joblib)
# se almacena localmente en ``ai_models/altur_detector_model.joblib``.
# En producción, también puede descargarse desde MinIO.
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_models"
    / "altur_detector_model.joblib"
)


class AudioDetector:
    """
    Detector de voz sintética basado en Wav2Vec2 + LogisticRegression.

    Ambos modos (streaming y lote) usan el MISMO modelo cargado en memoria;
    la diferencia está en cómo se alimenta:
    - Modo offline: WAV completo → ``detect_offline()``.
    - Modo streaming: chunks PCM → ``process_audio()``.

    Atributos:
        model      : clasificador LogisticRegression cargado de ``MODEL_PATH``.
        device     : dispositivo PyTorch (mps / cuda / cpu).
        processor  : tokenizador de Wav2Vec2.
        wav2vec    : modelo Wav2Vec2 preentrenado (facebook/wav2vec2-base).
        session    : sesión de streaming activa (DummySession).
    """

    def __init__(self):
        """
        Inicializa el detector cargando el modelo y Wav2Vec2.

        Selecciona automáticamente el mejor dispositivo disponible
        (MPS para Mac con Apple Silicon, CUDA para GPUs NVIDIA, CPU
        como fallback).
        """
        # Cargar el clasificador serializado.
        self.model = joblib.load(MODEL_PATH)

        # Seleccionar el mejor dispositivo de cómputo disponible.
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Cargando Wav2Vec2 en {self.device}...")

        # Cargar el tokenizador y el modelo Wav2Vec2 preentrenado.
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.wav2vec.eval()  # Modo inferencia (sin gradientes).

        # Crear la sesión de streaming para el WebSocket.
        self.session = DummySession(self)

    def _predict(self, audio_array: np.ndarray, sr: int) -> dict:
        """
        Pipeline interno de inferencia.

        Toma un array de audio mono (float32) y su sample rate, lo procesa
        a través de Wav2Vec2 y clasifica con el modelo entrenado.

        Pasos:
        1. Convertir a tensor PyTorch [1, num_samples].
        2. Resamplear a 16 kHz si es necesario.
        3. Truncar a 48,000 muestras (3 segundos a 16 kHz).
        4. Tokenizar con Wav2Vec2Processor.
        5. Forward pass → extraer capa oculta 4.
        6. Promedio temporal → vector de 768 dimensiones.
        7. Clasificar con LogisticRegression.

        Args:
            audio_array: Audio mono como numpy array float32.
            sr:          Sample rate del audio.

        Returns:
            Diccionario con el resultado de la detección.
        """
        # Convertir numpy a tensor PyTorch.
        waveform = torch.from_numpy(audio_array).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, num_samples]

        # Resamplear a 16 kHz si el audio tiene otra frecuencia.
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # Truncar a 3 segundos (48,000 muestras) para uniformidad.
        if waveform.shape[1] > 48000:
            waveform = waveform[:, :48000]

        # Tokenizar el audio para Wav2Vec2.
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass sin gradientes (inferencia).
        with torch.no_grad():
            outputs = self.wav2vec(**inputs, output_hidden_states=True)
            # Usar la capa oculta 4 (buen balance entre acústica y semántica).
            hidden_states = outputs.hidden_states[4]
            # Promedio temporal: [1, T, 768] → [768].
            features = hidden_states.mean(dim=1).squeeze().cpu().numpy()

        # Clasificar con LogisticRegression.
        features = features.reshape(1, -1)  # [1, 768]
        proba = self.model.predict_proba(features)[0][1]  # P(sintético)
        is_synthetic = bool(proba > 0.5)

        return {
            "t": 0.0,
            "is_synthetic": is_synthetic,
            # Confianza: qué tan seguro está el modelo de su predicción.
            "confidence": float(proba) if is_synthetic else float(1.0 - proba),
            # Score total es la probabilidad de la clase sintética.
            "score_total": float(proba),
            # LLR acústico acumulado (en este pipeline, igual al score).
            "llr_acoustic_cum": float(proba),
            # LLR comportamental (no implementado en este pipeline).
            "llr_behavioral_cum": 0.0,
            # Cantidad de segmentos acústicos analizados (1 por llamada offline).
            "n_acoustic_segments": 1,
            # Eventos comportamentales (no implementado en este pipeline).
            "n_behavioral_events": 0,
            # Umbral de decisión del clasificador.
            "eta": 0.5,
        }

    def process_audio(
        self,
        audio: bytes,
        sample_rate: int,
    ) -> dict | None:
        """
        Procesa un chunk de audio en modo streaming.

        Convierte los bytes PCM 16-bit mono a flotantes normalizados
        y los alimenta a la sesión de streaming. Si se han acumulado
        ~3 segundos de audio, ejecuta la predicción y devuelve el snapshot.

        Args:
            audio:       Bytes PCM 16-bit mono a 16 kHz.
            sample_rate: Frecuencia de muestreo (debe ser 16000 Hz).

        Returns:
            Diccionario con el snapshot si hay suficiente audio, None si no.

        Raises:
            ValueError: si sample_rate no es 16000 Hz.
        """
        if not audio:
            return None

        if sample_rate != 16_000:
            raise ValueError(
                "AudioDetector expects 16000 Hz audio."
            )

        # Convertir PCM 16-bit a flotantes normalizados [-1, 1].
        audio_array = np.frombuffer(
            audio,
            dtype=np.int16,
        ).astype(np.float64) / 32768.0

        # Alimentar la sesión de streaming.
        snapshot = self.session.push_audio_chunk(
            audio_array,
            sample_rate,
        )

        return snapshot

    def detect_offline(
        self,
        audio: bytes,
    ) -> dict:
        """
        Procesa una llamada completa en modo offline (batch).

        Lee el archivo WAV desde los bytes crudos, extrae el canal 0
        (llamante) si es estéreo, y ejecuta el pipeline de predicción.

        Args:
            audio: Bytes crudos del archivo WAV completo.

        Returns:
            Diccionario con el resultado de la detección.

        Raises:
            ValueError: si el audio está vacío.
        """
        if not audio:
            raise ValueError("Audio cannot be empty.")

        # Leer el WAV con soundfile (soporta múltiples formatos PCM).
        data, sr = sf.read(io.BytesIO(audio), dtype='float32')

        # Si es estéreo, quedarse con el canal 0 (llamante).
        if data.ndim > 1:
            data = data[:, 0]

        return self._predict(data, sr)


class DummySession:
    """
    Sesión de streaming simplificada para el WebSocket.

    Acumula chunks de audio en un buffer interno. Cuando se alcanzan
    ~3 segundos de audio (``sample_rate * 3`` muestras), concatena
    todo el buffer, ejecuta la predicción y retorna el snapshot.

    Atributos:
        parent    : instancia de AudioDetector (para acceder a ``_predict``).
        buffer    : lista de arrays numpy con los chunks acumulados.
        last_snap : último snapshot de detección generado.
    """

    def __init__(self, parent: AudioDetector):
        """
        Args:
            parent: instancia del AudioDetector padre.
        """
        self.parent = parent
        self.buffer: list[np.ndarray] = []
        self.last_snap: dict | None = None

    def push_audio_chunk(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> dict | None:
        """
        Agrega un chunk al buffer y ejecuta predicción si hay ~3 segundos.

        Args:
            audio_array: Array numpy con las muestras del chunk.
            sample_rate: Frecuencia de muestreo del audio.

        Returns:
            Snapshot de detección si se alcanzó el umbral, None si no.
        """
        self.buffer.append(audio_array)

        # Verificar si hemos acumulado ~3 segundos de audio.
        total_samples = sum(len(x) for x in self.buffer)
        if total_samples >= sample_rate * 3:
            # Concatenar todos los chunks y ejecutar la predicción.
            combined = np.concatenate(self.buffer)
            self.buffer = []  # Vaciar el buffer.
            self.last_snap = self.parent._predict(combined, sample_rate)
            return self.last_snap

        return None

    def current_snapshot(self) -> dict | None:
        """
        Devuelve el último snapshot de detección generado, o None
        si no se ha acumulado suficiente audio todavía.
        """
        return self.last_snap
