#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# All defaults below match RL_minimal.py (optimal for ICNN adversarial training).
# Override any flag via extra CLI args, e.g.:
#   bash run_RL_minimal_icnn.sh --env swingup_pendulum --iters 300
#   bash run_RL_minimal_icnn.sh --icnn-softplus-beta 10.0 --k-icnn 15
#
# --method accepts any subset of {nominal, ro, particle, icnn, algo1, npf,
#   ppa, new_ppa, dual, wgf, wfr, svg, rgo, nn_dro} plus the groups {both, all}.

exec "${PYTHON_BIN}" RL_minimal.py \
  --method nominal particle icnn \
  --grad-method pathwise \
  --env cartpole \
  --seed 0 \
  --iters 100 \
  --lam 3.0 \
  --k-icnn 10 \
  --eta-icnn 0.05 \
  --icnn-hidden-sizes 1024 512 512 256 128 64 \
  --icnn-init identity \
  --icnn-nonneg-init principled \
  --icnn-softplus-beta 20.0 \
  --save-gif --gif-every 10 --no-plot --gif-max-steps 500 \
  --gif-fps 20 \
  --save-json \
  "$@"
