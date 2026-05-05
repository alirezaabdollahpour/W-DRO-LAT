#!/usr/bin/env bash
# Train RL_minimal.py once per (seed, lam) cell, with one invocation producing
# a checkpoint for each method in $METHODS (default: the seven paper-table
# rows — ERM=nominal, PA=particle, WFR=wfr, SDRO=dual, NN-DRO=nn_dro,
# MPA=new_ppa, ICNN-DRO=npf). Each checkpoint persists the trained parametric
# adversary state (icnn / npf / nn_dro), which monge_gap_sweep.py's
# rl_cartpole backend reads back to apply the *trained* transport map
# T_psi(z_hat) directly when computing the gap.
#
# Outputs go to RL/monge_gap_runs/horizon_${HORIZON}/ (per-horizon subdir so
# checkpoints from different horizons don't collide). Override defaults via
# env vars or CLI passthrough:
#
#   bash run_RL_train_for_monge_gap.sh                       # default grid
#   LAMS="0.1 0.5 1.0 2.0 5.0" SEEDS="0 1 2" bash run_RL_train_for_monge_gap.sh
#   METHODS="nominal particle new_ppa npf" bash run_RL_train_for_monge_gap.sh
#   HORIZON=1000 bash run_RL_train_for_monge_gap.sh         # stress-test beyond 500
#   HORIZON=2000 FD_HORIZON=500 bash run_RL_train_for_monge_gap.sh
#   bash run_RL_train_for_monge_gap.sh --iters 100 --env cartpole
#
# HORIZON sets the env-max-steps (training rollout cap) AND the eval horizons
# (J_nominal / J_worst-grid measurements during training). Default 500 matches
# CartPole-v1's standard cap. Use larger HORIZON to stress-test whether
# methods continue to perform or fail at long horizons.
#
# FD_HORIZON sets the *inner-adversary* rollout length (used by particle, wfr,
# dual, new_ppa, npf, icnn, nn_dro for their inner ascent gradient estimates).
# It is independent of HORIZON: at HORIZON=2000 you still want a usable
# adversary signal but each rollout becomes 4x more expensive, so you may
# want to keep FD_HORIZON modest. Default = min(HORIZON, 200) (keeps inner
# loop cost flat for HORIZON >= 200; scales down for shorter envs).
#
# Anything past the script's positional args is forwarded verbatim to
# RL_minimal.py (after the standard hyperparameters).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

LAMS_DEFAULT="3.0 5.0"
SEEDS_DEFAULT="0"
ENV_DEFAULT="cartpole"
# Paper-table methods only (matches LR-CIFAR10 row set):
#   ERM=nominal, PA=particle, WFR=wfr, SDRO=dual, NN-DRO=nn_dro,
#   MPA=new_ppa, ICNN-DRO=npf  (npf, NOT older icnn; new_ppa, NOT older ppa)
METHODS_DEFAULT="nominal particle wfr dual nn_dro new_ppa npf"

LAMS="${LAMS:-${LAMS_DEFAULT}}"
SEEDS="${SEEDS:-${SEEDS_DEFAULT}}"
METHODS="${METHODS:-${METHODS_DEFAULT}}"
ENV_NAME="${ENV_NAME:-${ENV_DEFAULT}}"
HORIZON="${HORIZON:-500}"
# Validate HORIZON is a positive integer. The arithmetic comparison below
# relies on -lt, which silently aborts under set -e on non-integer input.
if ! [[ "${HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[run] ERROR: HORIZON must be a positive integer (got: '${HORIZON}')." >&2
  exit 1
fi
# FD_HORIZON: inner-adversary rollout length. Default min(HORIZON, 200).
if [ -z "${FD_HORIZON:-}" ]; then
  if [ "${HORIZON}" -lt 200 ]; then
    FD_HORIZON="${HORIZON}"
  else
    FD_HORIZON="200"
  fi
fi
if ! [[ "${FD_HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[run] ERROR: FD_HORIZON must be a positive integer (got: '${FD_HORIZON}')." >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/monge_gap_runs/horizon_${HORIZON}}"
mkdir -p "${OUT_DIR}"

echo "[run] LAMS=${LAMS}"
echo "[run] SEEDS=${SEEDS}"
echo "[run] METHODS=${METHODS}"
echo "[run] ENV=${ENV_NAME}"
echo "[run] HORIZON=${HORIZON}  FD_HORIZON=${FD_HORIZON}"
echo "[run] OUT_DIR=${OUT_DIR}"

for seed in ${SEEDS}; do
  for lam in ${LAMS}; do
    json_path="${OUT_DIR}/RL_minimal_${ENV_NAME}_paper_seed${seed}_lam_${lam}_npf_softplusbeta_10.0.json"
    ckpt_prefix="${json_path%.json}"
    # Skip a cell only if every requested method already has a checkpoint
    # for this exact run profile (so partial-failure cells get retried, and
    # older NPF profiles do not mask the CIFAR-aligned NPF checkpoints).
    all_present=1
    for m in ${METHODS}; do
      if [ ! -f "${ckpt_prefix}_${m}_policy.pt" ]; then
        all_present=0
        break
      fi
    done
    if [ "${all_present}" -eq 1 ]; then
      echo "[skip] seed=${seed} lam=${lam} (checkpoints for all methods already exist)"
      continue
    fi
    echo "[train] seed=${seed} lam=${lam} methods=${METHODS}"
    "${PYTHON_BIN}" RL_minimal.py \
      --method ${METHODS} \
      --grad-method pathwise \
      --env "${ENV_NAME}" \
      --seed "${seed}" \
      --iters 100 \
      --lam "${lam}" \
      --env-max-steps "${HORIZON}" \
      --eval-nom-horizon "${HORIZON}" \
      --eval-worst-horizon "${HORIZON}" \
      --fd-horizon "${FD_HORIZON}" \
      --k-icnn 10 --eta-icnn 0.05 \
      --icnn-hidden-sizes 1024 512 512 256 128 64 \
      --icnn-init identity --icnn-nonneg-init principled --icnn-softplus-beta 20.0 \
      --k-algo1 10 --lr-algo1 0.05 \
      --ppa-num-rounds 3 --ppa-inner-steps-round0 10 --ppa-inner-lr-round0 0.05 \
      --new-ppa-num-rounds 3 --new-ppa-inner-steps-round0 10 --new-ppa-inner-lr-round0 0.05 \
      --dual-epsilon 0.01 --dual-sample-level 3 \
      --particle-num-samples 8 --particle-inner-steps 10 --particle-inner-lr 0.05 --particle-epsilon 0.01 \
      --rgo-num-samples 8 --rgo-inner-steps 10 --rgo-inner-lr 0.05 --rgo-epsilon 0.01 --rgo-max-trials 10 \
      --k-npf 20 \
      --npf-hidden-sizes 512 512 256 128 64 \
      --npf-outer-rank 8 --npf-inner-rank 2 \
      --npf-activation softplus --npf-init-eps 1e-4 --npf-strong-convexity 1.0 --npf-softplus-beta 10.0 \
      --npf-eta 0.0002 --npf-bb-alpha-min 1e-7 --npf-bb-alpha-max 0.25 \
      --npf-bb-ls-c 1e-4 --npf-bb-ls-shrink 0.5 --npf-bb-ls-max-steps 15 \
      --k-nn-dro 10 --lr-nn-dro 0.05 \
      --nn-dro-hidden-sizes 256 128 64 --nn-dro-activation relu --nn-dro-init-scale 1e-3 \
      --no-plot --save-json \
      --json-path "${json_path}" \
      "$@"
  done
done

echo "[run] done. Outputs under ${OUT_DIR}"
