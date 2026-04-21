#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# Run all MNIST_Cuturi baselines + the two original WDRO methods in xi-space
# on the swing-up pendulum. xi = (m, l, b) (mass, length, damping).
# Override the method list or any flag from the CLI, e.g.:
#   bash run_RL_minimal_cuturi_baselines_swingup_pendulum.sh --method nn_dro
#   bash run_RL_minimal_cuturi_baselines_swingup_pendulum.sh --iters 200
#
# Methods: nominal, particle, icnn, algo1, npf, ppa, new_ppa, dual, wgf, wfr, svg, rgo, nn_dro.

exec "${PYTHON_BIN}" RL_minimal.py \
  --method all \
  --grad-method pathwise \
  --env swingup_pendulum \
  --seed 0 \
  --iters 50 \
  --lam 2.0 \
  --pendulum-dt 0.1 --pendulum-u-max 8.0 --pendulum-max-speed 8.0 \
  --pendulum-theta-tol 0.2 --pendulum-vel-tol 1.0 --pendulum-actions 3 \
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
