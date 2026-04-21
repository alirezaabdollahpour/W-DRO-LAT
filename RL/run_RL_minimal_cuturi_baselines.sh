#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# Run all MNIST_Cuturi baselines + the two original WDRO methods in xi-space.
# Override the method list from the CLI, e.g.:
#   bash run_RL_minimal_cuturi_baselines.sh --method algo1 ppa wfr
#   bash run_RL_minimal_cuturi_baselines.sh --env swingup_pendulum
#
# Methods: nominal, particle, icnn, algo1, npf, ppa, new_ppa, dual, wgf, wfr, svg, rgo, nn_dro.

exec "${PYTHON_BIN}" RL_minimal.py \
  --method all \
  --grad-method pathwise \
  --env cartpole \
  --seed 0 \
  --iters 50 \
  --lam 0.5 \
  --k-icnn 10 --eta-icnn 0.05 \
  --icnn-hidden-sizes 1024 512 512 256 128 64 \
  --icnn-init identity --icnn-nonneg-init principled --icnn-softplus-beta 20.0 \
  --k-algo1 10 --lr-algo1 0.05 \
  --ppa-num-rounds 3 --ppa-inner-steps-round0 10 --ppa-inner-lr-round0 0.05 \
  --new-ppa-num-rounds 3 --new-ppa-inner-steps-round0 10 --new-ppa-inner-lr-round0 0.05 \
  --dual-epsilon 0.01 --dual-sample-level 3 \
  --particle-num-samples 8 --particle-inner-steps 10 --particle-inner-lr 0.05 --particle-epsilon 0.01 \
  --rgo-num-samples 8 --rgo-inner-steps 10 --rgo-inner-lr 0.05 --rgo-epsilon 0.01 --rgo-max-trials 10 \
  --k-npf 10 --lr-npf 0.05 \
  --k-nn-dro 10 --lr-nn-dro 0.05 \
  --nn-dro-hidden-sizes 256 128 64 --nn-dro-activation relu --nn-dro-init-scale 1e-3 \
  --no-plot --save-json \
  "$@"
