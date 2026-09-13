import re
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score


RUTA_DATASET = Path(
    "detectorsemantico/transcripciones_dataset.csv"
)

RUTA_MODELO = Path(
    "detectorsemantico/modelo_texto.pkl"
)


def normalizar_texto(texto):
    texto = re.sub(r'(\d\s*){3,}', ' SECUENCIANUM ', texto)
    texto = re.sub(r'\b\d\b', 'DIGITO', texto)
    return texto


dataset = pd.read_csv(
    RUTA_DATASET,
    encoding="utf-8-sig"
)

dataset["texto"] = (
    dataset["texto"]
    .fillna("")
    .astype(str)
)

dataset["texto"] = dataset["texto"].apply(normalizar_texto)


# Usamos TODOS los datos (train + val juntos) para el k-fold
X = dataset["texto"].reset_index(drop=True)
y = dataset["clase"].reset_index(drop=True)


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

    return Pipeline([
        ("tfidf", caracteristicas),
        (
            "clasificador",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])


kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

balanced_scores = []
auc_scores = []
inciertos_totales = 0

print()
print("Iniciando 5-fold cross-validation...")
print()

for fold, (idx_train, idx_val) in enumerate(kfold.split(X, y), start=1):

    X_train_fold = X.iloc[idx_train]
    y_train_fold = y.iloc[idx_train]

    X_val_fold = X.iloc[idx_val]
    y_val_fold = y.iloc[idx_val]

    modelo_fold = crear_modelo()
    modelo_fold.fit(X_train_fold, y_train_fold)

    predicciones = modelo_fold.predict(X_val_fold)
    probabilidades = modelo_fold.predict_proba(X_val_fold)[:, 1]

    ba = balanced_accuracy_score(y_val_fold, predicciones)
    auc = roc_auc_score(y_val_fold, probabilidades)

    incierto = ((probabilidades > 0.4) & (probabilidades < 0.6)).sum()

    balanced_scores.append(ba)
    auc_scores.append(auc)
    inciertos_totales += incierto

    print(
        f"Fold {fold}: Balanced Accuracy = {round(ba*100, 2)}%  "
        f"AUC = {round(auc, 4)}  "
        f"Inciertos = {incierto}/{len(X_val_fold)}"
    )


print()
print("RESULTADOS PROMEDIO (5-fold)")
print()
print("Balanced Accuracy:", round(np.mean(balanced_scores) * 100, 2), "% (+/-", round(np.std(balanced_scores) * 100, 2), ")")
print("AUC:", round(np.mean(auc_scores), 4))
print("Total casos inciertos:", inciertos_totales, "de", len(X))


# Modelo final: se entrena con TODOS los datos para producción
print()
print("Entrenando modelo final con todos los datos...")

modelo_final = crear_modelo()
modelo_final.fit(X, y)

joblib.dump(modelo_final, RUTA_MODELO)

print()
print("Modelo guardado en:")
print(RUTA_MODELO)