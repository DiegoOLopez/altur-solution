import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


RUTA_DATASET = Path(
    "detectorsemantico/transcripciones_dataset.csv"
)


dataset = pd.read_csv(
    RUTA_DATASET,
    encoding="utf-8-sig"
)

dataset["texto"] = (
    dataset["texto"]
    .fillna("")
    .astype(str)
)


train = dataset[
    dataset["split"] == "train"
].copy()

val = dataset[
    dataset["split"] == "val"
].copy()


X_train = train["texto"]
y_train = train["clase"]

X_val = val["texto"]
y_val = val["clase"]


def crear_modelo():

    caracteristicas = FeatureUnion([
        (
            "palabras",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
        ),

        (
            "caracteres",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
        )
    ])

    modelo = Pipeline([
        (
            "tfidf",
            caracteristicas
        ),

        (
            "clasificador",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return modelo


print()
print(
    "Buscando mejor umbral usando SOLO TRAIN..."
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


modelo_cv = crear_modelo()


probabilidades_train = cross_val_predict(
    modelo_cv,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]


mejor_umbral = 0.50
mejor_balanced = 0


for umbral in np.arange(
    0.30,
    0.71,
    0.01
):

    predicciones = (
        probabilidades_train >= umbral
    ).astype(int)

    balanced = balanced_accuracy_score(
        y_train,
        predicciones
    )

    if balanced > mejor_balanced:

        mejor_balanced = balanced
        mejor_umbral = umbral


print()
print(
    "Mejor umbral encontrado:",
    round(
        mejor_umbral,
        2
    )
)

print(
    "Balanced Accuracy CV TRAIN:",
    round(
        mejor_balanced * 100,
        2
    ),
    "%"
)


print()
print(
    "Entrenando modelo final con todo TRAIN..."
)


modelo_final = crear_modelo()

modelo_final.fit(
    X_train,
    y_train
)


probabilidades_val = modelo_final.predict_proba(
    X_val
)[:, 1]


predicciones_val = (
    probabilidades_val >= mejor_umbral
).astype(int)


balanced_val = balanced_accuracy_score(
    y_val,
    predicciones_val
)


print()
print(
    "RESULTADOS EN VAL"
)

print()

print(
    "Umbral usado:",
    round(
        mejor_umbral,
        2
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_val * 100,
        2
    ),
    "%"
)

print()

print(
    classification_report(
        y_val,
        predicciones_val,
        target_names=[
            "HUMANO",
            "SINTETICO"
        ]
    )
)

print(
    "Matriz de confusion:"
)

print(
    confusion_matrix(
        y_val,
        predicciones_val
    )
)