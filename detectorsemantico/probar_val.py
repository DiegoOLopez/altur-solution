import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


RUTA_DATASET = Path(
    "detectorsemantico/transcripciones_dataset.csv"
)

RUTA_MODELO = Path(
    "detectorsemantico/modelo_texto.pkl"
)


dataset = pd.read_csv(
    RUTA_DATASET,
    encoding="utf-8-sig"
)

modelo = joblib.load(
    RUTA_MODELO
)


val = dataset[
    dataset["split"] == "val"
].copy()


reales = []
predicciones = []


print()
print("PROBANDO LLAMADAS DE VALIDACION")
print()


for numero, (_, fila) in enumerate(
    val.iterrows(),
    start=1
):
    anon_id = fila["anon_id"]
    real = int(fila["clase"])
    texto = str(fila["texto"])

    probabilidades = modelo.predict_proba(
        [texto]
    )[0]

    prob_humano = float(
        probabilidades[0]
    )

    prob_sintetico = float(
        probabilidades[1]
    )

    if prob_sintetico >= 0.5:
        prediccion = 1
        confianza = prob_sintetico
    else:
        prediccion = 0
        confianza = prob_humano

    reales.append(
        real
    )

    predicciones.append(
        prediccion
    )

    real_texto = (
        "SINTETICO"
        if real == 1
        else "HUMANO"
    )

    pred_texto = (
        "SINTETICO"
        if prediccion == 1
        else "HUMANO"
    )

    correcto = (
        "SI"
        if real == prediccion
        else "NO"
    )

    print(
        f"{numero}/{len(val)}",
        anon_id
    )

    print(
        "Real:",
        real_texto
    )

    print(
        "Prediccion:",
        pred_texto
    )

    print(
        "Confianza:",
        round(
            confianza * 100,
            2
        ),
        "%"
    )

    print(
        "Correcto:",
        correcto
    )

    print()


balanced = balanced_accuracy_score(
    reales,
    predicciones
)


print("RESULTADO FINAL")
print()

print(
    "Balanced Accuracy:",
    round(
        balanced * 100,
        2
    ),
    "%"
)

print()

print(
    "Matriz de confusion:"
)

print(
    confusion_matrix(
        reales,
        predicciones
    )
)