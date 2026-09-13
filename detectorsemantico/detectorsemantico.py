import os
import re
import sys
import wave
import joblib
import numpy as np

from pathlib import Path
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel


sys.stdout.reconfigure(
    encoding="utf-8"
)


RUTA_MODELO = Path(
    "detectorsemantico/modelo_texto.pkl"
)


modelo_texto = joblib.load(
    RUTA_MODELO
)


print("Cargando Whisper...")

whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Whisper listo")


def normalizar_texto(texto):
    texto = re.sub(r'(\d\s*){3,}', ' SECUENCIANUM ', texto)
    texto = re.sub(r'\b\d\b', 'DIGITO', texto)
    return texto


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
            "El audio debe ser PCM de 16 bits"
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


def transcribir_audio(
    ruta_audio
):
    temporal = extraer_canal_caller(
        ruta_audio
    )

    try:
        segmentos, _ = whisper.transcribe(
            temporal,
            language="es",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        textos = []

        for segmento in segmentos:

            texto = segmento.text.strip()

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


def detectar_texto(
    texto
):
    texto = normalizar_texto(texto)

    probabilidades = modelo_texto.predict_proba(
        [texto]
    )[0]

    prob_humano = float(
        probabilidades[0]
    )

    prob_sintetico = float(
        probabilidades[1]
    )

    if prob_sintetico >= 0.5:

        clase = 1
        tipo = "SINTETICO"
        confianza = prob_sintetico

    else:

        clase = 0
        tipo = "HUMANO"
        confianza = prob_humano

    return {
        "clase": clase,
        "tipo": tipo,
        "confianza": confianza,
        "prob_humano": prob_humano,
        "prob_sintetico": prob_sintetico
    }


print()

ruta = input(
    "Pega la ruta del archivo WAV: "
).strip()

ruta = ruta.replace(
    '"',
    ""
)

ruta_audio = Path(
    ruta
)


if not ruta_audio.exists():

    print()
    print(
        "ERROR: No se encontro el archivo"
    )

    sys.exit()


texto = transcribir_audio(
    ruta_audio
)


resultado = detectar_texto(
    texto
)


print()
print(
    "RESULTADO"
)

print()

print(
    "Clase:",
    resultado["clase"]
)

print(
    "Confianza:",
    round(
        resultado["confianza"] * 100,
        2
    ),
    "%"
)