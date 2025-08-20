# GPUacceleration

Minimal, beginner-friendly GPU simulation for the SS-compliant UDE using DiffEqGPU. Training stays on CPU; this repo shows how to run many scenarios fast on GPU (with CPU fallback).

## What this does
- Loads SS-compliant dataset (`data/*_roadmap_correct.csv`).
- Packs signals into GPU arrays.
- Runs per-scenario ODEs with a GPU-safe RHS via DiffEqGPU Ensemble.
- Falls back to CPU automatically if GPU is unavailable.

## Quick start (Colab T4 or CPU)
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/gpu_ensemble_ude.jl
```

Outputs: `results/gpu_ensemble_summary.bson`

## Why another repo?
Your main pipeline trains UDE/BNode and evaluates on CPU. This repo demonstrates the correct GPU pattern: no DataFrames in `rhs!`, pre-packed arrays, and `DiffEqGPU` Ensembles.
