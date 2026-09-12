#!/usr/bin/env python3
"""
Genera un dataset de DEMO (audio sintetizado matemáticamente, no voces
reales) para poder correr todo el pipeline end-to-end sin depender del
dataset real de Altur. Sirve para probar que entrenamiento + inferencia +
API + dashboard funcionan antes de conectar el dataset verdadero.

Simula la diferencia H0 (humano) vs H1 (sintético) con:
  - H0: voz modelada como F0 con jitter/shimmer natural (ruido en periodo y
    amplitud ciclo a ciclo), formantes con ruido de fase, pausas de
    duración variable (respiración), latencias de reacción variables.
  - H1: tono más "limpio" (glotal casi periódico, jitter/shimmer bajos),
    fase más coherente entre frames, pausas muy uniformes, latencias de
    reacción post-turno del agente muy consistentes (baja varianza).

IMPORTANTE: esto es solo para *validar el codigo*. Para el reto real hay
que apuntar train/fit_densities.py al dataset real de Altur.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000  # debe coincidir con TARGET_SR en model/pipeline.py


def synth_voice(duration, sr, f0_base, jitter_amt, shimmer_amt, phase_noise, seed):
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    # F0 con jitter (perturbación ciclo a ciclo aproximada vía FM de baja frecuencia + ruido)
    f0_wave = f0_base * (1 + jitter_amt * rng.standard_normal(n).cumsum() / sr * 5)
    phase = 2 * np.pi * np.cumsum(f0_wave) / sr
    phase += phase_noise * rng.standard_normal(n)
    harmonics = sum(
        (1.0 / k) * np.sin(k * phase + rng.uniform(0, 0.1) * phase_noise)
        for k in range(1, 6)
    )
    # shimmer: perturbación de amplitud ciclo a ciclo (aprox. envolvente de baja frecuencia)
    amp_env = 1 + shimmer_amt * rng.standard_normal(n // 20 + 1).repeat(20)[:n]
    signal = harmonics * amp_env
    signal += 0.02 * rng.standard_normal(n)  # ruido de fondo de línea telefónica
    signal /= np.abs(signal).max() + 1e-9
    return signal * 0.8


def synth_silence(duration, sr):
    return np.zeros(int(duration * sr))


def build_call(label: str, seed: int, sr=SR):
    rng = np.random.default_rng(seed)
    is_synth = label == "synthetic"

    # --- parámetros de voz según hipótesis ---
    jitter_amt = 0.02 if is_synth else 0.15
    shimmer_amt = 0.01 if is_synth else 0.08
    phase_noise = 0.02 if is_synth else 0.25

    segments = []  # (channel, signal)
    turns = []
    t_cursor = 0.0

    n_exchanges = rng.integers(5, 8)
    for k in range(n_exchanges):
        # turno del agente
        agent_dur = rng.uniform(1.5, 3.5)
        segments.append((1, synth_voice(agent_dur, sr, 140, 0.15, 0.08, 0.2, seed * 100 + k)))
        turns.append({"channel": 1, "start": t_cursor, "end": t_cursor + agent_dur})
        agent_end = t_cursor + agent_dur
        t_cursor = agent_end

        # latencia de reacción del caller: consistente si es síntesis, variable si es humano
        if is_synth:
            latency = 0.35 + rng.normal(0, 0.02)  # muy consistente
        else:
            latency = max(0.1, rng.normal(0.6, 0.35))  # muy variable, a veces casi 0 (interrupción)
        latency = max(latency, -0.3)

        if latency > 0:
            segments.append((None, synth_silence(latency, sr)))
        t_cursor += latency

        caller_dur = rng.uniform(1.0, 3.0)
        f0_base = rng.uniform(100, 220)
        caller_start = t_cursor
        segments.append((0, synth_voice(caller_dur, sr, f0_base, jitter_amt, shimmer_amt, phase_noise, seed * 1000 + k)))
        turns.append({"channel": 0, "start": caller_start, "end": caller_start + caller_dur})
        t_cursor = caller_start + caller_dur

        gap = rng.uniform(0.2, 0.6)
        t_cursor += gap

    total_len = int(t_cursor * sr) + sr
    caller_track = np.zeros(total_len)
    agent_track = np.zeros(total_len)
    cursor = 0
    for ch, sig in segments:
        n = len(sig)
        if ch == 0:
            caller_track[cursor:cursor + n] += sig
        elif ch == 1:
            agent_track[cursor:cursor + n] += sig
        cursor += n

    stereo = np.stack([caller_track, agent_track], axis=1)
    return stereo, {"turns": turns}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="data/train")
    ap.add_argument("--n_per_class", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for label in ["human", "synthetic"]:
        folder = out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_per_class):
            seed = hash((label, i)) % (2**31)
            stereo, turns_json = build_call(label, seed)
            wav_path = folder / f"call{i:03d}.wav"
            sf.write(str(wav_path), stereo, SR)
            with open(wav_path.with_suffix("").with_suffix(".turns.json"), "w") as f:
                json.dump(turns_json, f)
        print(f"[✔] {args.n_per_class} llamadas de demo generadas en {folder}")


if __name__ == "__main__":
    main()
