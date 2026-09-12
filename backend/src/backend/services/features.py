import numpy as np

from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window

from scipy.fft import dct


def extract_basic_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    if audio.dtype != np.float32:
        raise ValueError(
            f"Expected float32 audio, received {audio.dtype}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, received {sample_rate} Hz."
        )

    if audio.size == 0:
        raise ValueError("Audio segment is empty.")

    peak_amplitude = float(np.max(np.abs(audio)))

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio, dtype=np.float64)
            )
        )
    )

    zero_crossings = np.sum(
        np.diff(np.signbit(audio))
    )

    zero_crossing_rate = float(
        zero_crossings / max(audio.size - 1, 1)
    )

    energy = float(
        np.sum(
            np.square(audio, dtype=np.float64)
        )
    )

    frame_size = int(0.025 * sample_rate)

    if audio.size >= frame_size:
        frame_count = audio.size // frame_size

        trimmed = audio[:frame_count * frame_size]

        frames = trimmed.reshape(
            frame_count,
            frame_size,
        )

        frame_rms = np.sqrt(
            np.mean(
                np.square(frames, dtype=np.float64),
                axis=1,
            )
        )

        energy_mean = float(np.mean(frame_rms))
        energy_std = float(np.std(frame_rms))
    else:
        energy_mean = rms
        energy_std = 0.0

    return {
        "peak_amplitude": peak_amplitude,
        "rms": rms,
        "zero_crossing_rate": zero_crossing_rate,
        "energy": energy,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }

def extract_segment(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int,
) -> np.ndarray:
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    start_sample = max(start_sample, 0)
    end_sample = min(end_sample, audio.size)

    if end_sample <= start_sample:
        raise ValueError(
            "Invalid speech segment."
        )

    return audio[start_sample:end_sample]


def extract_spectral_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    if audio.dtype != np.float32:
        raise ValueError(
            f"Expected float32 audio, received {audio.dtype}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, received {sample_rate} Hz."
        )

    if audio.size == 0:
        raise ValueError("Audio segment is empty.")

    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.010 * sample_rate)

    if audio.size < frame_size:
        audio = np.pad(
            audio,
            (0, frame_size - audio.size),
        )

    window = get_window(
        "hann",
        frame_size,
    )

    frequencies = rfftfreq(
        frame_size,
        1 / sample_rate,
    )

    centroids = []
    bandwidths = []
    rolloffs = []
    flatness_values = []
    flux_values = []

    previous_spectrum = None

    for start in range(
        0,
        audio.size - frame_size + 1,
        hop_size,
    ):
        frame = audio[start:start + frame_size]

        spectrum = np.abs(
            rfft(frame * window)
        )

        power = spectrum ** 2

        total_power = np.sum(power)

        if total_power <= 0:
            continue

        centroid = np.sum(
            frequencies * power
        ) / total_power

        bandwidth = np.sqrt(
            np.sum(
                ((frequencies - centroid) ** 2) * power
            ) / total_power
        )

        cumulative_power = np.cumsum(power)
        threshold = 0.85 * total_power

        rolloff_index = np.searchsorted(
            cumulative_power,
            threshold,
        )

        rolloff_index = min(
            rolloff_index,
            len(frequencies) - 1,
        )

        rolloff = frequencies[rolloff_index]

        geometric_mean = np.exp(
            np.mean(
                np.log(spectrum + 1e-10)
            )
        )

        arithmetic_mean = np.mean(
            spectrum
        )

        flatness = (
            geometric_mean /
            max(arithmetic_mean, 1e-10)
        )

        normalized_spectrum = (
            spectrum /
            np.sum(spectrum)
        )

        if previous_spectrum is None:
            flux = 0.0
        else:
            flux = np.sqrt(
                np.sum(
                    (
                        normalized_spectrum -
                        previous_spectrum
                    ) ** 2
                )
            )

        previous_spectrum = normalized_spectrum

        centroids.append(centroid)
        bandwidths.append(bandwidth)
        rolloffs.append(rolloff)
        flatness_values.append(flatness)
        flux_values.append(flux)

    if not centroids:
        raise ValueError(
            "Unable to extract spectral features."
        )

    return {
        "spectral_centroid_mean": float(
            np.mean(centroids)
        ),
        "spectral_centroid_std": float(
            np.std(centroids)
        ),
        "spectral_centroid_min": float(
            np.min(centroids)
        ),
        "spectral_centroid_max": float(
            np.max(centroids)
        ),
        "spectral_bandwidth_mean": float(
            np.mean(bandwidths)
        ),
        "spectral_bandwidth_std": float(
            np.std(bandwidths)
        ),
        "spectral_rolloff_mean": float(
            np.mean(rolloffs)
        ),
        "spectral_rolloff_std": float(
            np.std(rolloffs)
        ),
        "spectral_flatness_mean": float(
            np.mean(flatness_values)
        ),
        "spectral_flatness_std": float(
            np.std(flatness_values)
        ),
        "spectral_flux_mean": float(
            np.mean(flux_values)
        ),
        "spectral_flux_std": float(
            np.std(flux_values)
        ),
    }

def extract_mfcc_features(
    audio: np.ndarray,
    sample_rate: int,
    num_mfcc: int = 13,
) -> dict[str, float]:

    if audio.dtype != np.float32:
        raise ValueError(
            f"Expected float32 audio, received {audio.dtype}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, received {sample_rate} Hz."
        )

    if audio.size == 0:
        raise ValueError("Audio segment is empty.")

    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.010 * sample_rate)
    n_fft = 512

    if audio.size < frame_size:
        audio = np.pad(
            audio,
            (0, frame_size - audio.size),
        )

    window = get_window(
        "hann",
        frame_size,
    )

    mel_filterbank = _create_mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        num_filters=26,
    )

    mfcc_frames = []

    for start in range(
        0,
        audio.size - frame_size + 1,
        hop_size,
    ):
        frame = audio[start:start + frame_size]

        spectrum = np.abs(
            rfft(frame * window, n=n_fft)
        )

        power_spectrum = (
            spectrum ** 2
        )

        mel_energies = (
            mel_filterbank @ power_spectrum
        )

        mel_energies = np.maximum(
            mel_energies,
            1e-10,
        )

        log_mel = np.log(
            mel_energies
        )

        mfcc = dct(
            log_mel,
            type=2,
            norm="ortho",
        )[:num_mfcc]

        mfcc_frames.append(mfcc)

    if not mfcc_frames:
        raise ValueError(
            "Unable to extract MFCC features."
        )

    mfcc_frames = np.asarray(
        mfcc_frames,
        dtype=np.float32,
    )

    delta = _compute_delta(mfcc_frames)
    delta_delta = _compute_delta(delta)

    features = {}

    _add_statistics(
        features,
        "mfcc",
        mfcc_frames,
    )

    _add_statistics(
        features,
        "delta_mfcc",
        delta,
    )

    _add_statistics(
        features,
        "delta_delta_mfcc",
        delta_delta,
    )

    return features


def _hz_to_mel(frequency: float) -> float:
    return 2595.0 * np.log10(
        1.0 + frequency / 700.0
    )


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def _create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    num_filters: int,
) -> np.ndarray:
    min_frequency = 0.0
    max_frequency = sample_rate / 2

    min_mel = _hz_to_mel(
        min_frequency
    )

    max_mel = _hz_to_mel(
        max_frequency
    )

    mel_points = np.linspace(
        min_mel,
        max_mel,
        num_filters + 2,
    )

    frequencies = _mel_to_hz(
        mel_points
    )

    bins = np.floor(
        (n_fft + 1) * frequencies / sample_rate
    ).astype(int)

    filterbank = np.zeros(
        (num_filters, n_fft // 2 + 1)
    )

    for index in range(num_filters):
        left = bins[index]
        center = bins[index + 1]
        right = bins[index + 2]

        if center > left:
            for frequency_bin in range(
                left,
                center,
            ):
                if frequency_bin < filterbank.shape[1]:
                    filterbank[
                        index,
                        frequency_bin,
                    ] = (
                        frequency_bin - left
                    ) / (center - left)

        if right > center:
            for frequency_bin in range(
                center,
                right,
            ):
                if frequency_bin < filterbank.shape[1]:
                    filterbank[
                        index,
                        frequency_bin,
                    ] = (
                        right - frequency_bin
                    ) / (right - center)

    return filterbank

def _compute_delta(
    features: np.ndarray,
) -> np.ndarray:
    delta = np.zeros_like(features)

    if features.shape[0] == 1:
        return delta

    delta[1:-1] = (
        features[2:] - features[:-2]
    ) / 2.0

    delta[0] = (
        features[1] - features[0]
    )

    delta[-1] = (
        features[-1] - features[-2]
    )

    return delta


def _add_statistics(
    output: dict[str, float],
    prefix: str,
    values: np.ndarray,
) -> None:
    for index in range(values.shape[1]):
        coefficient = values[:, index]

        output[
            f"{prefix}_{index + 1:02d}_mean"
        ] = float(np.mean(coefficient))

        output[
            f"{prefix}_{index + 1:02d}_std"
        ] = float(np.std(coefficient))

        output[
            f"{prefix}_{index + 1:02d}_min"
        ] = float(np.min(coefficient))

        output[
            f"{prefix}_{index + 1:02d}_max"
        ] = float(np.max(coefficient))

def extract_pitch_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    if audio.dtype != np.float32:
        raise ValueError(
            f"Expected float32 audio, received {audio.dtype}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, received {sample_rate} Hz."
        )

    if audio.size == 0:
        raise ValueError("Audio segment is empty.")

    frame_size = int(0.040 * sample_rate)
    hop_size = int(0.010 * sample_rate)

    min_frequency = 70.0
    max_frequency = 400.0

    min_lag = int(sample_rate / max_frequency)
    max_lag = int(sample_rate / min_frequency)

    window = get_window(
        "hann",
        frame_size,
    )

    pitch_values = []

    for start in range(
        0,
        audio.size - frame_size + 1,
        hop_size,
    ):
        frame = audio[start:start + frame_size]

        frame = frame - np.mean(frame)
        frame = frame * window

        energy = np.sum(frame ** 2)

        if energy <= 1e-8:
            continue

        autocorrelation = np.correlate(
            frame,
            frame,
            mode="full",
        )

        autocorrelation = autocorrelation[
            frame_size - 1:
        ]

        search_end = min(
            max_lag,
            len(autocorrelation),
        )

        if search_end <= min_lag:
            continue

        search_region = autocorrelation[
            min_lag:search_end
        ]

        peak_index = np.argmax(
            search_region
        )

        lag = min_lag + peak_index

        peak_value = autocorrelation[lag]

        if peak_value <= 0:
            continue

        normalized_peak = (
            peak_value / autocorrelation[0]
            )

        if normalized_peak < 0.3:
            continue

        frequency = sample_rate / lag

        if min_frequency <= frequency <= max_frequency:
            pitch_values.append(frequency)

    if not pitch_values:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "f0_range": 0.0,
            "f0_voiced_ratio": 0.0,
        }

    pitch_values = np.asarray(
        pitch_values,
        dtype=np.float32,
    )

    return {
        "f0_mean": float(
            np.mean(pitch_values)
        ),
        "f0_std": float(
            np.std(pitch_values)
        ),
        "f0_min": float(
            np.min(pitch_values)
        ),
        "f0_max": float(
            np.max(pitch_values)
        ),
        "f0_range": float(
            np.max(pitch_values)
            - np.min(pitch_values)
        ),
        "f0_voiced_ratio": float(
            len(pitch_values)
            / max(
                1,
                int(
                    max(
                        1,
                        (
                            audio.size
                            - frame_size
                        )
                        // hop_size
                        + 1
                    )
                ),
            )
        ),
    }


def extract_voice_quality_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    if audio.dtype != np.float32:
        raise ValueError("Audio must be float32.")

    if sample_rate != 16000:
        raise ValueError("Sample rate must be 16000 Hz.")

    if audio.size == 0:
        raise ValueError("Audio cannot be empty.")

    frame_size = int(0.04 * sample_rate)
    hop_size = int(0.01 * sample_rate)

    if audio.size < frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))

    periods = []
    amplitudes = []
    harmonic_ratios = []

    previous_period = None
    previous_amplitude = None

    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 70)

    for start in range(0, audio.size - frame_size + 1, hop_size):
        frame = audio[start:start + frame_size]

        frame = frame - np.mean(frame)

        rms = float(np.sqrt(np.mean(np.square(frame))))

        if rms < 0.005:
            continue

        autocorrelation = np.correlate(
            frame,
            frame,
            mode="full",
        )

        autocorrelation = autocorrelation[len(autocorrelation) // 2:]

        if autocorrelation[0] <= 0:
            continue

        max_valid_lag = min(max_lag, len(autocorrelation) - 1)

        if min_lag >= max_valid_lag:
            continue

        search_region = autocorrelation[min_lag:max_valid_lag + 1]

        peak_index = int(np.argmax(search_region))
        lag = min_lag + peak_index

        correlation = (
            autocorrelation[lag] / autocorrelation[0]
        )

        if correlation < 0.3:
            continue

        period = lag / sample_rate

        periods.append(period)
        amplitudes.append(rms)

        harmonic_ratio = max(float(correlation), 1e-10)
        harmonic_ratios.append(harmonic_ratio)

        if previous_period is not None:
            period_variation = abs(period - previous_period)
        else:
            period_variation = 0.0

        if previous_amplitude is not None:
            amplitude_variation = abs(rms - previous_amplitude)
        else:
            amplitude_variation = 0.0

        previous_period = period
        previous_amplitude = rms

    if not periods:
        return {
            "jitter_mean": 0.0,
            "jitter_std": 0.0,
            "shimmer_mean": 0.0,
            "shimmer_std": 0.0,
            "hnr_mean": 0.0,
            "hnr_std": 0.0,
        }

    periods_array = np.asarray(periods, dtype=np.float64)
    amplitudes_array = np.asarray(amplitudes, dtype=np.float64)
    harmonic_array = np.asarray(harmonic_ratios, dtype=np.float64)

    if len(periods_array) > 1:
        jitter_values = (
            np.abs(np.diff(periods_array))
            / np.maximum(periods_array[:-1], 1e-10)
        )
    else:
        jitter_values = np.array([0.0])

    if len(amplitudes_array) > 1:
        shimmer_values = (
            np.abs(np.diff(amplitudes_array))
            / np.maximum(amplitudes_array[:-1], 1e-10)
        )
    else:
        shimmer_values = np.array([0.0])

    hnr_values = 10.0 * np.log10(
        harmonic_array / np.maximum(1.0 - harmonic_array, 1e-10)
    )

    return {
        "jitter_mean": float(np.mean(jitter_values)),
        "jitter_std": float(np.std(jitter_values)),
        "shimmer_mean": float(np.mean(shimmer_values)),
        "shimmer_std": float(np.std(shimmer_values)),
        "hnr_mean": float(np.mean(hnr_values)),
        "hnr_std": float(np.std(hnr_values)),
    }


def extract_acoustic_features(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int,
) -> dict[str, float]:
    segment_audio = extract_segment(
        audio,
        start,
        end,
        sample_rate,
    )

    basic_features = extract_basic_features(
        segment_audio,
        sample_rate,
    )

    spectral_features = extract_spectral_features(
        segment_audio,
        sample_rate,
    )

    mfcc_features = extract_mfcc_features(
        segment_audio,
        sample_rate,
    )

    pitch_features = extract_pitch_features(
        segment_audio,
        sample_rate,
    )

    voice_quality_features = extract_voice_quality_features(
        segment_audio,
        sample_rate,
    )

    features = {
        "duration": float(end - start),
        **basic_features,
        **spectral_features,
        **mfcc_features,
        **pitch_features,
        **voice_quality_features,
    }

    return features

ACOUSTIC_FEATURE_NAMES = [
    "duration",
    "peak_amplitude",
    "rms",
    "zero_crossing_rate",
    "energy",
    "energy_mean",
    "energy_std",

    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_centroid_min",
    "spectral_centroid_max",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_flux_mean",
    "spectral_flux_std",
]

for prefix in ("mfcc", "delta_mfcc", "delta_delta_mfcc"):
    for index in range(1, 14):
        for statistic in ("mean", "std", "min", "max"):
            ACOUSTIC_FEATURE_NAMES.append(
                f"{prefix}_{index:02d}_{statistic}"
            )

ACOUSTIC_FEATURE_NAMES.extend([
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "f0_range",
    "f0_voiced_ratio",

    "jitter_mean",
    "jitter_std",
    "shimmer_mean",
    "shimmer_std",
    "hnr_mean",
    "hnr_std",
])


VOICE_EMBEDDING_FEATURE_NAMES = [
    f"voice_embedding_{index:03d}"
    for index in range(1, 193)
]


MODEL_FEATURE_NAMES = (
    ACOUSTIC_FEATURE_NAMES
    + VOICE_EMBEDDING_FEATURE_NAMES
)

def build_model_matrix(
    acoustic_vectors: list[dict],
    voice_embedding_segments: list[dict],
) -> np.ndarray:
    if not acoustic_vectors:
        raise ValueError("Acoustic vectors cannot be empty.")

    if not voice_embedding_segments:
        raise ValueError(
            "Voice embedding segments cannot be empty."
        )

    if len(acoustic_vectors) != len(
        voice_embedding_segments
    ):
        raise ValueError(
            "Acoustic and voice embedding segment counts "
            "must match."
        )

    rows = []

    for acoustic_item, embedding_item in zip(
        acoustic_vectors,
        voice_embedding_segments,
    ):
        acoustic_features = np.asarray(
            [
                acoustic_item["features"][name]
                for name in ACOUSTIC_FEATURE_NAMES
            ],
            dtype=np.float32,
        )

        voice_embedding = np.asarray(
            embedding_item["embedding"],
            dtype=np.float32,
        )

        if voice_embedding.size != len(
            VOICE_EMBEDDING_FEATURE_NAMES
        ):
            raise ValueError(
                "Unexpected voice embedding dimension: "
                f"{voice_embedding.size}. "
                f"Expected "
                f"{len(VOICE_EMBEDDING_FEATURE_NAMES)}."
            )

        row = np.concatenate(
            [
                acoustic_features,
                voice_embedding,
            ]
        )

        rows.append(row)

    matrix = np.stack(rows).astype(np.float32)

    if matrix.shape[1] != len(MODEL_FEATURE_NAMES):
        raise ValueError(
            "Model matrix feature count does not match "
            "the feature name count."
        )

    return matrix

def build_model_features(
    model_matrix: np.ndarray,
    speech_segments: list[tuple[float, float]],
    feature_names: list[str],
    call_embedding_mean: np.ndarray,
    call_embedding_std: np.ndarray,
    call_embedding_mean_norm: float,
    call_embedding_std_norm: float,
) -> dict:
    if model_matrix.dtype != np.float32:
        raise ValueError("Model matrix must be float32.")

    if model_matrix.ndim != 2:
        raise ValueError("Model matrix must be two-dimensional.")

    if model_matrix.shape[0] != len(speech_segments):
        raise ValueError(
            "Model matrix rows must match speech segments."
        )

    if model_matrix.shape[1] != len(feature_names):
        raise ValueError(
            "Model matrix columns must match feature names."
        )

    if call_embedding_mean.size != 192:
        raise ValueError(
            "Call mean embedding must have 192 dimensions."
        )

    if call_embedding_std.size != 192:
        raise ValueError(
            "Call std embedding must have 192 dimensions."
        )

    segments = []

    for index, ((start, end), feature_vector) in enumerate(
        zip(speech_segments, model_matrix),
        start=1,
    ):
        segments.append({
            "segment": index,
            "start": float(start),
            "end": float(end),
            "duration": float(end - start),
            "feature_vector": feature_vector.tolist(),
        })

    return {
        "sample_rate": 16000,
        "feature_names": feature_names,
        "feature_matrix": model_matrix.tolist(),
        "segments": segments,
        "call_embedding": {
            "mean": call_embedding_mean.tolist(),
            "std": call_embedding_std.tolist(),
            "mean_norm": float(call_embedding_mean_norm),
            "std_norm": float(call_embedding_std_norm),
        },
    }

def aggregate_model_features(
    model_matrix: np.ndarray,
) -> np.ndarray:
    if model_matrix.dtype != np.float32:
        raise ValueError("Model matrix must be float32.")

    if model_matrix.ndim != 2:
        raise ValueError("Model matrix must be two-dimensional.")

    if model_matrix.shape[0] == 0:
        raise ValueError("Model matrix cannot be empty.")

    mean_features = np.mean(
        model_matrix,
        axis=0,
    )

    std_features = np.std(
        model_matrix,
        axis=0,
    )

    min_features = np.min(
        model_matrix,
        axis=0,
    )

    max_features = np.max(
        model_matrix,
        axis=0,
    )

    aggregated_features = np.concatenate([
        mean_features,
        std_features,
        min_features,
        max_features,
    ])

    return aggregated_features.astype(np.float32)