import sys
import wave
import os
import numpy as np
import pandas as pd

from pathlib import Path
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel


sys.stdout.reconfigure(
    encoding="utf-8"
)


RUTA_MANIFEST = Path(
    r"C:\Users\Usuario\Documents\hackmty26\manifest.csv"
)

RUTA_HUMANOS = Path(
    r"C:\Users\Usuario\Documents\hackmty26\audio_16k\train16\human16"
)

RUTA_SINTETICOS = Path(
    r"C:\Users\Usuario\Documents\hackmty26\audio_16k\train16\synthetic16"
)

RUTA_SALIDA = Path(
    "detectorsemantico/transcripciones_dataset.csv"
)


print()
print("Cargando Whisper...")

modelo = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Modelo listo")


def buscar_audio(
    anon_id,
    label
):
    if label == "human":
        return (
            RUTA_HUMANOS
            / f"{anon_id}.wav"
        )

    return (
        RUTA_SINTETICOS
        / f"{anon_id}.wav"
    )


def extraer_canal_caller(
    ruta_audio
):
    with wave.open(
        str(ruta_audio),
        "rb"
    ) as wav:

        canales = wav.getnchannels()
        frecuencia = wav.getframerate()
        ancho = wav.getsampwidth()
        frames = wav.getnframes()

        datos = wav.readframes(
            frames
        )

    if ancho != 2:
        raise ValueError(
            "El audio no es PCM de 16 bits"
        )

    audio = np.frombuffer(
        datos,
        dtype=np.int16
    )

    if canales == 2:
        caller = audio[0::2]

    elif canales == 1:
        caller = audio

    else:
        raise ValueError(
            f"Numero de canales no soportado: {canales}"
        )

    temporal = NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temporal.close()

    with wave.open(
        temporal.name,
        "wb"
    ) as salida:

        salida.setnchannels(1)
        salida.setsampwidth(2)
        salida.setframerate(
            frecuencia
        )

        salida.writeframes(
            caller.astype(
                np.int16
            ).tobytes()
        )

    return temporal.name


def limpiar_texto(
    texto
):
    texto = texto.strip()

    texto = " ".join(
        texto.split()
    )

    return texto


def transcribir_audio(
    ruta_audio
):
    temporal = extraer_canal_caller(
        ruta_audio
    )

    try:
        segmentos, _ = modelo.transcribe(
            temporal,
            language="es",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        textos = []

        for segmento in segmentos:
            texto = limpiar_texto(
                segmento.text
            )

            if texto:
                textos.append(
                    texto
                )

        return " ".join(
            textos
        )

    finally:
        if os.path.exists(
            temporal
        ):
            os.remove(
                temporal
            )


if not RUTA_MANIFEST.exists():
    print()
    print(
        "ERROR: No se encontro manifest.csv"
    )
    print(
        RUTA_MANIFEST
    )
    sys.exit()


manifest = pd.read_csv(
    RUTA_MANIFEST,
    encoding="utf-8"
)


columnas_necesarias = {
    "anon_id",
    "label",
    "split"
}


if not columnas_necesarias.issubset(
    manifest.columns
):
    print()
    print(
        "ERROR: El manifest no tiene las columnas necesarias"
    )

    print(
        manifest.columns.tolist()
    )

    sys.exit()


resultados = []

total = len(
    manifest
)


print()
print(
    "Total de llamadas:",
    total
)


for numero, (_, fila) in enumerate(
    manifest.iterrows(),
    start=1
):
    anon_id = str(
        fila["anon_id"]
    ).strip()

    label = str(
        fila["label"]
    ).strip()

    split = str(
        fila["split"]
    ).strip()

    ruta_audio = buscar_audio(
        anon_id,
        label
    )

    print()
    print(
        f"{numero}/{total}",
        anon_id,
        label,
        split
    )

    if not ruta_audio.exists():
        print(
            "NO ENCONTRADO:"
        )

        print(
            ruta_audio
        )

        continue

    try:
        texto = transcribir_audio(
            ruta_audio
        )

        if label == "synthetic":
            clase = 1

        elif label == "human":
            clase = 0

        else:
            print(
                "Label desconocido:",
                label
            )

            continue

        resultados.append({
            "anon_id": anon_id,
            "label": label,
            "clase": clase,
            "split": split,
            "texto": texto
        })

        dataset = pd.DataFrame(
            resultados
        )

        dataset.to_csv(
            RUTA_SALIDA,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "Texto:"
        )

        print(
            texto[:300]
        )

    except Exception as error:
        print()
        print(
            "ERROR:"
        )

        print(
            anon_id
        )

        print(
            error
        )


print()
print(
    "PROCESO TERMINADO"
)

print()

print(
    "Transcripciones guardadas en:"
)

print(
    RUTA_SALIDA
)

print()

print(
    "Total procesadas:",
    len(resultados)
)