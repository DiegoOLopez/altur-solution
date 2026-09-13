"""
Aumento de datos (augmentación) a nivel de forma de onda — SOLO para el
split de entrenamiento de cada fold, nunca para validación.

Idea central y su justificación:

Con ~350 llamadas reales, el cuello de botella no es "más épocas" (este
modelo no usa descenso de gradiente por lotes) sino la VARIANZA de la
estimación: pocas llamadas por clase implican gaussianas y pesos de
stacking sensibles a qué llamadas concretas cayeron en cada split. La
augmentación ataca esto por el lado de los datos: en vez de inventar
llamadas falsas (contenido nuevo, hablantes nuevos, texto nuevo — lo cual
sí introduciría información que no existe), se generan variantes
realistas de canal/entorno de las MISMAS llamadas reales que ya se
etiquetaron. Esto es exactamente lo que le pasa a un mismo audio real
cuando cambia la codificación de la línea, el ruido de fondo, el nivel de
grabación o la red: la llamada sigue siendo la misma persona diciendo lo
mismo, pero el detector debe seguir funcionando.

Reglas de diseño (importantes, no accesorias):

1. Solo se perturban condiciones de CANAL/ENTORNO (ruido, ancho de banda
   telefónico, códec, pérdida de paquetes, ganancia, reverberación leve).
   Deliberadamente NO se usa time-stretch ni pitch-shift agresivo: esas
   transformaciones alterarían jitter/shimmer y la coherencia de fase
   armónica, que son justo las features que distinguen voz humana de
   sintética — augmentar así podría borrar la señal que se quiere
   detectar, no solo diversificar el ruido de fondo.

2. Cada variante aumentada se etiqueta con el mismo `group_id` (id de la
   llamada original) que el audio real del que proviene. Esto importa
   para dos cosas del resto del pipeline:
     - Nunca puede quedar una variante aumentada de una llamada en el
       fold de VALIDACIÓN mientras el audio original de esa misma llamada
       está en TRAIN (o viceversa): la augmentación se genera después del
       split, únicamente para las llamadas ya asignadas a train de ese
       fold/repetición.
     - El shrinkage de varianza por llamada (`model/density.py`) sigue
       viendo estas variantes como parte de la MISMA llamada (mismo
       group_id), no como llamadas independientes nuevas — si se contaran
       como independientes, la varianza "entre llamadas" se subestimaría
       artificialmente, exactamente el sesgo que ese shrinkage existe para
       evitar.

3. Todas las transformaciones son deterministas dado un `numpy.random.Generator`
   con semilla fija, para que el entrenamiento sea reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# Transformaciones individuales (todas reciben/regresan float64, misma escala
# de amplitud que la señal de entrada — no se asume ningún rango fijo).
# ---------------------------------------------------------------------------

def add_gaussian_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido blanco aditivo a una relación señal/ruido dada (simula línea
    ruidosa / micrófono de baja calidad)."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def telephone_bandpass(signal: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    """Filtro pasa-banda tipo telefonía fija (300-3400 Hz), simula la
    limitación de ancho de banda de muchas líneas/códecs de voz."""
    signal = np.asarray(signal, dtype=np.float64)
    nyq = sr / 2.0
    high = min(high_hz, nyq * 0.99)
    if signal.size < 32 or low_hz >= high:
        return signal
    sos = butter(4, [low_hz / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def mu_law_roundtrip(signal: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Compansión mu-law (codifica y decodifica), simula el códec G.711
    usado en telefonía VoIP — introduce cuantización no lineal leve."""
    signal = np.asarray(signal, dtype=np.float64)
    peak = np.abs(signal).max()
    if peak < 1e-9:
        return signal
    x = np.clip(signal / peak, -1.0, 1.0)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    decoded = np.sign(encoded) * ((1 + mu) ** np.abs(encoded) - 1) / mu
    return decoded * peak


def random_gain_db(signal: np.ndarray, low_db: float, high_db: float, rng: np.random.Generator) -> np.ndarray:
    """Cambio de nivel de grabación (ganancia) aleatorio dentro de un rango."""
    gain_db = rng.uniform(low_db, high_db)
    return np.asarray(signal, dtype=np.float64) * (10 ** (gain_db / 20.0))


def packet_dropout(signal: np.ndarray, sr: int, rng: np.random.Generator,
                    loss_rate: float = 0.02, burst_ms: tuple[float, float] = (20.0, 60.0)) -> np.ndarray:
    """Simula pérdida de paquetes de una llamada VoIP: silencia ráfagas
    cortas y esporádicas del audio."""
    signal = np.asarray(signal, dtype=np.float64).copy()
    n = signal.size
    if n == 0:
        return signal
    step = max(int(sr * 0.02), 1)  # candidatos a "paquete" cada 20ms
    pos = 0
    while pos < n:
        if rng.random() < loss_rate:
            burst_len = int(sr * rng.uniform(*burst_ms) / 1000.0)
            signal[pos:pos + burst_len] = 0.0
            pos += burst_len
        else:
            pos += step
    return signal


def light_reverb(signal: np.ndarray, sr: int, rng: np.random.Generator,
                  decay_ms: tuple[float, float] = (40.0, 140.0)) -> np.ndarray:
    """Reverberación leve vía una respuesta al impulso sintética de
    decaimiento exponencial (sin bancos de IR externos), simula una sala o
    un manos-libres en vez de un micrófono de cerca."""
    signal = np.asarray(signal, dtype=np.float64)
    dur_ms = rng.uniform(*decay_ms)
    n_ir = max(int(sr * dur_ms / 1000.0), 4)
    t = np.arange(n_ir)
    tau = n_ir / 3.0
    ir = np.exp(-t / tau) * rng.normal(1.0, 0.15, size=n_ir)
    ir[0] = 1.0  # conserva el sonido directo dominante
    ir = ir / np.sum(np.abs(ir))
    wet = np.convolve(signal, ir, mode="full")[:signal.size]
    mix = rng.uniform(0.15, 0.35)  # reverbe leve, nunca domina la señal directa
    return (1 - mix) * signal + mix * wet


def _safety_limit(signal: np.ndarray, ref_peak: float) -> np.ndarray:
    """Evita overflow numérico tras encadenar transformaciones, sin alterar
    la dinámica salvo en el caso extremo de saturación."""
    if ref_peak <= 1e-9:
        return signal
    cur_peak = np.abs(signal).max()
    if cur_peak > ref_peak * 4.0:
        signal = signal * (ref_peak * 4.0 / cur_peak)
    return np.clip(signal, -ref_peak * 6.0, ref_peak * 6.0)


# ---------------------------------------------------------------------------
# Perfiles combinados: cada uno representa una condición de canal plausible.
# ---------------------------------------------------------------------------

def _profile_noise_low_snr(s, sr, rng):
    return add_gaussian_noise(s, rng.uniform(8.0, 15.0), rng)


def _profile_noise_mid_snr(s, sr, rng):
    return add_gaussian_noise(s, rng.uniform(18.0, 28.0), rng)


def _profile_telephone(s, sr, rng):
    return telephone_bandpass(s, sr)


def _profile_codec(s, sr, rng):
    return mu_law_roundtrip(s)


def _profile_packet_loss(s, sr, rng):
    return packet_dropout(s, sr, rng)


def _profile_gain(s, sr, rng):
    return random_gain_db(s, -7.0, 7.0, rng)


def _profile_reverb(s, sr, rng):
    return light_reverb(s, sr, rng)


def _profile_phone_plus_noise(s, sr, rng):
    return telephone_bandpass(add_gaussian_noise(s, rng.uniform(12.0, 22.0), rng), sr)


def _profile_codec_plus_dropout(s, sr, rng):
    return packet_dropout(mu_law_roundtrip(s), sr, rng, loss_rate=0.015)


def _profile_reverb_plus_gain(s, sr, rng):
    return random_gain_db(light_reverb(s, sr, rng), -4.0, 4.0, rng)


AUGMENTATION_PROFILES = [
    ("noise_low_snr", _profile_noise_low_snr),
    ("noise_mid_snr", _profile_noise_mid_snr),
    ("telephone_narrowband", _profile_telephone),
    ("codec_mulaw", _profile_codec),
    ("packet_loss", _profile_packet_loss),
    ("gain_shift", _profile_gain),
    ("reverb_light", _profile_reverb),
    ("phone_plus_noise", _profile_phone_plus_noise),
    ("codec_plus_dropout", _profile_codec_plus_dropout),
    ("reverb_plus_gain", _profile_reverb_plus_gain),
]


def generate_augmented_variants(signal: np.ndarray, sr: int, rng: np.random.Generator,
                                 n_variants: int) -> list[tuple[str, np.ndarray]]:
    """Regresa hasta `n_variants` variantes (tag, señal_aumentada) de la
    MISMA llamada, cada una con una condición de canal distinta. Si
    `n_variants` supera el número de perfiles disponibles, se repiten
    perfiles con nuevas realizaciones aleatorias (nunca se inventa
    contenido nuevo, solo más muestras de la misma condición)."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or n_variants <= 0:
        return []
    ref_peak = float(np.abs(signal).max()) if signal.size else 0.0

    order = rng.permutation(len(AUGMENTATION_PROFILES))
    variants = []
    for k in range(n_variants):
        name, fn = AUGMENTATION_PROFILES[order[k % len(AUGMENTATION_PROFILES)]]
        aug = fn(signal, sr, rng)
        aug = _safety_limit(np.asarray(aug, dtype=np.float64), ref_peak)
        tag = name if k < len(AUGMENTATION_PROFILES) else f"{name}_v{k // len(AUGMENTATION_PROFILES) + 1}"
        variants.append((tag, aug))
    return variants
