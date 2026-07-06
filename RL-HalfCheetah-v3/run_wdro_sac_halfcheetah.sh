#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# Defaults mirror the CIFAR NPF-LastQuad run style. Override any value with
# environment variables or by passing CLI flags after the script name.
export ENV_ID="${ENV_ID:-HalfCheetah-v5}"
export RUN_ONLY_ALGO="${RUN_ONLY_ALGO:-npf_lastquad}"
export PENALTY_LAMBDA="${PENALTY_LAMBDA:-30}"
export COMMON_BATCH="${COMMON_BATCH:-512}"
export NPF_LASTQUAD_HIDDEN="${NPF_LASTQUAD_HIDDEN:-128 128 64 32}"
export NPF_LASTQUAD_ACTIVATION="${NPF_LASTQUAD_ACTIVATION:-softplus}"
export NPF_LASTQUAD_SOFTPLUS_BETA="${NPF_LASTQUAD_SOFTPLUS_BETA:-5.0}"
export NPF_LASTQUAD_INIT_EPS="${NPF_LASTQUAD_INIT_EPS:-1e-4}"
export NPF_LASTQUAD_STRONG_CONVEXITY="${NPF_LASTQUAD_STRONG_CONVEXITY:-1.0}"
export OMEGA_STEPS="${OMEGA_STEPS:-20}"
export INP_STEPS="${INP_STEPS:-20}"
export INP_RESTARTS="${INP_RESTARTS:-5}"
export BB_ALPHA0="${BB_ALPHA0:-1e-3}"
export BB_ALPHA_MIN="${BB_ALPHA_MIN:-1e-6}"
export BB_ALPHA_MAX="${BB_ALPHA_MAX:-0.05}"
export BB_LS_C="${BB_LS_C:-1e-5}"
export BB_LS_SHRINK="${BB_LS_SHRINK:-0.5}"
export BB_LS_MAX_STEPS="${BB_LS_MAX_STEPS:-20}"

exec "${PYTHON_BIN}" train_sac_wdro.py "$@"
