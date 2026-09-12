from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np

from backend.schemas.detection import DetectionResponse
from backend.schemas.model_features import ModelFeatures

from backend.services.audio import process_wav
from backend.services.channel import extract_channel0
from backend.services.vad import detect_speech_segments

from backend.services.features import (
    extract_segment,
    extract_acoustic_features,
    build_model_matrix,
    build_model_features,
    aggregate_model_features,
    MODEL_FEATURE_NAMES,
)

from backend.services.embeddings import (
    extract_voice_embedding,
    aggregate_voice_embeddings,
)


router = APIRouter(
    prefix="/detect",
    tags=["Detection"],
)


@router.post("", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Recibe un archivo WAV, lo normaliza a 16 kHz,
    extrae el canal 0, detecta segmentos de voz
    y obtiene las características acústicas de cada segmento.
    """

    # Validamos que el archivo tenga nombre.
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required.",
        )

    # Validamos que sea un archivo WAV.
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported.",
        )

    # Leemos el archivo.
    audio_bytes = await file.read()

    # Validamos que no esté vacío.
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        # Procesamos el WAV.
        #
        # Si está a 8 kHz, lo convertimos a 16 kHz.
        (
            original_sample_rate,
            final_sample_rate,
            converted,
            processed_audio,
        ) = process_wav(audio_bytes)

        # Extraemos únicamente el canal 0.
        #
        # Este canal corresponde a la voz del caller.
        channel0, _ = extract_channel0(processed_audio)

        print(
            f"Channel 0: {len(channel0)} samples "
            f"({len(channel0) / final_sample_rate:.2f} seconds)"
        )
        print(f"Channel 0 dtype: {channel0.dtype}")
        print(f"Channel 0 min: {channel0.min():.6f}")
        print(f"Channel 0 max: {channel0.max():.6f}")

        # Detectamos los segmentos donde existe voz.
        speech_segments = detect_speech_segments(
            channel0,
            final_sample_rate,
        )

        print(
            f"Final speech segments: "
            f"{len(speech_segments)}"
        )

        for index, (start, end) in enumerate(
            speech_segments,
            start=1,
        ):
            print(
                f"Segment {index}: "
                f"{start:.2f}s - {end:.2f}s "
                f"({end - start:.2f}s)"
            )

        # Aquí guardaremos las características
        # acústicas y los embeddings.
        acoustic_vectors = []
        voice_embeddings = []
        voice_embedding_segments = []

        for index, (start, end) in enumerate(
            speech_segments,
            start=1,
        ):
            # Extraemos el audio correspondiente
            # únicamente a este segmento.
            segment_audio = extract_segment(
                channel0,
                start,
                end,
                final_sample_rate,
            )

            # Extraemos las 187 características acústicas.
            acoustic_features = extract_acoustic_features(
                channel0,
                start,
                end,
                final_sample_rate,
            )

            # Extraemos el embedding ECAPA de 192 dimensiones.
            voice_embedding = extract_voice_embedding(
                segment_audio,
                final_sample_rate,
            )

            voice_embeddings.append(voice_embedding)

            # Guardamos el embedding junto con
            # la información temporal del segmento.
            voice_embedding_segments.append({
                "segment": index,
                "start": start,
                "end": end,
                "duration": end - start,
                "embedding": voice_embedding,
            })

            # Guardamos las características acústicas
            # junto con la información temporal.
            acoustic_vectors.append({
                "segment": index,
                "start": start,
                "end": end,
                "features": acoustic_features,
            })

            print(
                f"Voice embedding segment {index}: "
                f"dimension={len(voice_embedding)}, "
                f"norm={np.linalg.norm(voice_embedding):.4f}"
            )

            print(
                f"Acoustic vector segment {index}: "
                f"{len(acoustic_features)} features"
            )

        # Calculamos la duración de cada segmento.
        durations = [
            end - start
            for start, end in speech_segments
        ]

        # Creamos un embedding global de la llamada.
        voice_embedding_summary = aggregate_voice_embeddings(
            voice_embeddings,
            durations,
        )

        # Construimos la matriz final.
        #
        # Cada fila = un segmento.
        # Cada fila = 379 características.
        #
        # 187 acústicas + 192 ECAPA = 379.
        model_matrix = build_model_matrix(
            acoustic_vectors,
            voice_embedding_segments,
        )


        # Convertimos la cantidad variable de segmentos
        # en un vector de tamaño fijo.
        aggregated_features = aggregate_model_features(
            model_matrix
        )

        print(
            f"Aggregated feature vector size: "
            f"{len(aggregated_features)}"
        )

        print(
            f"Aggregated feature vector dtype: "
            f"{aggregated_features.dtype}"
        )

        # Convertimos la matriz y la información
        # de la llamada en una estructura organizada.
        model_features_data = build_model_features(
            model_matrix=model_matrix,
            speech_segments=speech_segments,
            feature_names=MODEL_FEATURE_NAMES,
            call_embedding_mean=(
                voice_embedding_summary["mean_embedding"]
            ),
            call_embedding_std=(
                voice_embedding_summary["std_embedding"]
            ),
            call_embedding_mean_norm=(
                voice_embedding_summary["mean_norm"]
            ),
            call_embedding_std_norm=(
                voice_embedding_summary["std_norm"]
            ),
        )

        # Validamos la estructura con Pydantic.
        #
        # Aquí comprobamos que todos los datos tienen
        # exactamente el formato que definimos en
        # schemas/model_features.py.
        model_features = ModelFeatures.model_validate(
            model_features_data
        )

        print(
            f"ModelFeatures validated: "
            f"{type(model_features).__name__}"
        )

        # Número de segmentos.
        print(
            f"ModelFeatures segments: "
            f"{len(model_features.segments)}"
        )

        # Número total de características.
        print(
            f"ModelFeatures feature names: "
            f"{len(model_features.feature_names)}"
        )

        # Número de características de cada segmento.
        print(
            f"ModelFeatures vector size: "
            f"{len(model_features.segments[0].feature_vector)}"
        )

        # Número de filas de la matriz.
        print(
            f"ModelFeatures matrix rows: "
            f"{len(model_features.feature_matrix)}"
        )

        # Número de columnas de la matriz.
        print(
            f"ModelFeatures matrix columns: "
            f"{len(model_features.feature_matrix[0])}"
        )

        # Comprobamos que podemos convertir
        # toda la estructura a un formato JSON.
        model_features_json = model_features.model_dump(
            mode="json"
        )


        # Verificamos las partes principales del JSON.
        print(
            f"JSON sample rate: "
            f"{model_features_json['sample_rate']}"
        )

        print(
            f"JSON feature matrix rows: "
            f"{len(model_features_json['feature_matrix'])}"
        )

        print(
            f"JSON feature matrix columns: "
            f"{len(model_features_json['feature_matrix'][0])}"
        )

        print(
            f"JSON segments: "
            f"{len(model_features_json['segments'])}"
        )

        print(
            f"JSON call embedding mean: "
            f"{len(model_features_json['call_embedding']['mean'])}"
        )

        print(
            f"JSON call embedding std: "
            f"{len(model_features_json['call_embedding']['std'])}"
        )

        print(
            f"ModelFeatures JSON-ready: "
            f"{isinstance(model_features_json, dict)}"
        )

        # Información adicional de validación.
        print(
            f"First feature: "
            f"{MODEL_FEATURE_NAMES[0]}"
        )

        print(
            f"Last acoustic feature: "
            f"{MODEL_FEATURE_NAMES[186]}"
        )

        print(
            f"First embedding feature: "
            f"{MODEL_FEATURE_NAMES[187]}"
        )

        print(
            f"Last feature: "
            f"{MODEL_FEATURE_NAMES[378]}"
        )

        print(
            f"Model matrix shape: "
            f"{model_matrix.shape}"
        )

        print(
            f"Model matrix dtype: "
            f"{model_matrix.dtype}"
        )

        print(
            f"Voice embedding segments stored: "
            f"{len(voice_embedding_segments)}"
        )

        print(
            "Voice embedding dimensions: "
            f"{len(voice_embedding_segments[0]['embedding'])}"
        )

        print(
            "Call voice embedding: "
            f"dimension="
            f"{len(voice_embedding_summary['mean_embedding'])}"
        )

        print(
            "Call embedding norm: "
            f"mean={voice_embedding_summary['mean_norm']:.4f}, "
            f"std={voice_embedding_summary['std_norm']:.4f}"
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Por ahora NO modificamos la respuesta de /detect.
    #
    # ModelFeatures se está construyendo y validando internamente,
    # pero todavía no lo enviamos al frontend.
    return DetectionResponse(
        message=(
            "Audio converted from 8000 Hz to 16000 Hz."
            if converted
            else "Audio was already 16000 Hz. "
                 "No conversion was necessary."
        ),
        original_sample_rate=original_sample_rate,
        final_sample_rate=final_sample_rate,
        converted=converted,
    )