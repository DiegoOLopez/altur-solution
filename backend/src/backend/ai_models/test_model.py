from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).parent / "model.pkl"
)


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