from pathlib import Path
import sys

import joblib


# Obtenemos la raíz del repositorio.
# test_model.py está en:
#
# backend/src/backend/ai_models/test_model.py
#
# parents[4] nos lleva a:
#
# altur-solution/
REPO_ROOT = Path(__file__).resolve().parents[4]

# El código original del modelo está en:
#
# altur-solution/detector/
DETECTOR_ROOT = REPO_ROOT / "detector"

# Agregamos la carpeta detector al PYTHONPATH
# para que Python pueda encontrar módulos como:
#
# model.pipeline
# model.density
# model.calibration
if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))


# Ruta del modelo entrenado.
MODEL_PATH = (
    Path(__file__).resolve().parent / "altur_detector_model.pkl"
)


# Cargamos el modelo.
model = joblib.load(MODEL_PATH)


print("Tipo:")
print(type(model))

print("\nVersión:")
print(getattr(model, "VERSION", None))

print("\nSegment seconds:")
print(getattr(model, "segment_seconds", None))

print("\nAcoustic block:")
print(type(model.acoustic_block))

print("\nBehavioral block:")
print(type(model.behavioral_block))

print("\nCalibrator:")
print(type(model.calibrator))

print("\nETA:")
print(model.eta)

print("\nPrior H1:")
print(model.prior_h1)

print("\nMétodos:")
print("new_session:", hasattr(model, "new_session"))

print(
    "predict_offline:",
    hasattr(model, "predict_offline"),
)