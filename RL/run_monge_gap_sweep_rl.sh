#!/usr/bin/env bash
# Run monge_gap_sweep.py against the RL CartPole checkpoints produced by
# run_RL_train_for_monge_gap.sh, then emit a LaTeX table mirroring the
# LR-CIFAR10 paper layout (tab:monge_gap_vs_lambda_lr).
#
# Location-independent: works whether this script lives at the repo root
# (LAT/) or inside LAT/RL/. It auto-detects REPO_ROOT by looking for
# monge_gap_sweep.py in the script's directory and its parent.
#
# Defaults:
#   - HORIZON = 500 (matches CartPole-v1's standard episode cap).
#   - CKPT_DIR = ${REPO_ROOT}/RL/monge_gap_runs/horizon_${HORIZON}/
#     (per-horizon subdir produced by run_RL_train_for_monge_gap.sh).
#   - OUT_CSV / OUT_TEX placed alongside the checkpoints.
#   - METHODS = paper-table rows
#       ERM=nominal, PA=particle, WFR=wfr, SDRO=dual,
#       NN-DRO=nn_dro, MPA=new_ppa, ICNN-DRO=npf.
#   - EVAL_SIZE = 4096 (statistically tight in 2D xi-space; tractable for
#     non-parametric pushes; bypasses new_ppa's O(B^2) memory cliff).
#   - GAP_MODE = sinkhorn (1000 iters, eps adaptive at 1% of median pairwise
#     sq-distance — matches the paper caption verbatim).
#
# Override:
#   LAMS="0.1 0.5 1.0 2.0 5.0" SEEDS="0 1 2" bash run_monge_gap_sweep_rl.sh
#   HORIZON=1000 bash run_monge_gap_sweep_rl.sh        # eval at horizon 1000
#   METHODS="nominal particle new_ppa npf" bash run_monge_gap_sweep_rl.sh
#   EVAL_SIZE=8192 GAP_MODE=hungarian bash run_monge_gap_sweep_rl.sh
#
# Stress-test pattern (eval pre-trained policies at a longer horizon):
#   HORIZON=2000 \
#   CKPT_DIR=.../horizon_500 \
#   OUT_CSV=.../stress_h500_eval2000.csv \
#   OUT_TEX=.../stress_h500_eval2000.tex \
#   bash run_monge_gap_sweep_rl.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Locate REPO_ROOT (the directory containing monge_gap_sweep.py). This
# script can live either at the repo root or inside RL/; pick whichever
# parent has the sweep entrypoint. Failing to detect is a hard error so
# the user can't accidentally run a misplaced copy.
if [ -f "${SCRIPT_DIR}/monge_gap_sweep.py" ]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/../monge_gap_sweep.py" ]; then
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
else
  echo "[sweep] ERROR: could not find monge_gap_sweep.py in ${SCRIPT_DIR} or ${SCRIPT_DIR}/.." >&2
  echo "[sweep]        Place this script at the repo root or inside RL/." >&2
  exit 1
fi
RL_DIR="${REPO_ROOT}/RL"
TABLE_PY="${RL_DIR}/RL_monge_gap_table.py"

# All Python invocations run from REPO_ROOT so monge_gap_sweep.py's relative
# imports (Logistic_Regression_CIFAR10/, RL/) resolve.
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"

HORIZON="${HORIZON:-500}"
if ! [[ "${HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[sweep] ERROR: HORIZON must be a positive integer (got: '${HORIZON}')." >&2
  exit 1
fi
# FD_HORIZON: inner-adversary rollout length used by non-parametric methods
# during push (particle, wfr, dual, new_ppa). Independent of HORIZON.
if [ -z "${FD_HORIZON:-}" ]; then
  if [ "${HORIZON}" -lt 200 ]; then
    FD_HORIZON="${HORIZON}"
  else
    FD_HORIZON="200"
  fi
fi
if ! [[ "${FD_HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[sweep] ERROR: FD_HORIZON must be a positive integer (got: '${FD_HORIZON}')." >&2
  exit 1
fi

CKPT_DIR="${CKPT_DIR:-${RL_DIR}/monge_gap_runs/horizon_${HORIZON}}"
OUT_CSV="${OUT_CSV:-${CKPT_DIR}/monge_gap_rl_cartpole.csv}"
OUT_TEX="${OUT_TEX:-${CKPT_DIR}/monge_gap_rl_cartpole.tex}"

LAMS="${LAMS:-0.1 0.5 1.0 2.0 5.0}"
SEEDS="${SEEDS:-0 1 2}"
# Paper-table methods. MPA row uses new_ppa (multi-start particle ascent
# with batch-wide reassignment, Algorithm 1 in the paper). ICNN-DRO row uses
# npf (NPF / Vesseron-Cuturi), matching the LR-CIFAR10 paper convention.
METHODS="${METHODS:-nominal particle wfr dual nn_dro new_ppa npf}"
EVAL_SIZE="${EVAL_SIZE:-4096}"
ENV_NAME="${ENV_NAME:-cartpole}"
GAP_MODE="${GAP_MODE:-sinkhorn}"
# Sinkhorn precision knobs. Defaults match the paper (1% of median pairwise
# sq-distance, 1000 log-domain iterations). Override SINKHORN_EPS_FRAC=0.05
# for tighter convergence at the cost of more entropic bias (canceled by
# the debiased Sinkhorn divergence). The solver is log-domain so any eps
# is numerically safe.
SINKHORN_EPS_FRAC="${SINKHORN_EPS_FRAC:-0.01}"
SINKHORN_N_ITER="${SINKHORN_N_ITER:-1000}"

# Validate the checkpoint directory before launching a long sweep — the
# Python backend's per-cell error is opaque ("No RL policy found for ...").
if [ ! -d "${CKPT_DIR}" ]; then
  echo "[sweep] ERROR: CKPT_DIR=${CKPT_DIR} does not exist." >&2
  echo "[sweep]        Run run_RL_train_for_monge_gap.sh first (with the same HORIZON)." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"

echo "[sweep] REPO_ROOT=${REPO_ROOT}"
echo "[sweep] CKPT_DIR=${CKPT_DIR}"
echo "[sweep] METHODS=${METHODS}"
echo "[sweep] LAMS=${LAMS}"
echo "[sweep] SEEDS=${SEEDS}"
echo "[sweep] HORIZON=${HORIZON}  FD_HORIZON=${FD_HORIZON}"
echo "[sweep] EVAL_SIZE=${EVAL_SIZE}  GAP_MODE=${GAP_MODE}"
echo "[sweep] OUT_CSV=${OUT_CSV}"

"${PYTHON_BIN}" "${REPO_ROOT}/monge_gap_sweep.py" \
  --dataset rl_cartpole \
  --methods ${METHODS} \
  --seeds ${SEEDS} \
  --lams ${LAMS} \
  --eval_size "${EVAL_SIZE}" \
  --gap_mode "${GAP_MODE}" \
  --sinkhorn_eps_frac "${SINKHORN_EPS_FRAC}" \
  --sinkhorn_n_iter "${SINKHORN_N_ITER}" \
  --rl_env "${ENV_NAME}" \
  --rl_env_max_steps "${HORIZON}" \
  --rl_fd_horizon "${FD_HORIZON}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --out "${OUT_CSV}" \
  "$@"

"${PYTHON_BIN}" "${TABLE_PY}" \
  --csv "${OUT_CSV}" \
  --out "${OUT_TEX}" \
  --lams ${LAMS} \
  --methods ${METHODS}

echo "[sweep] wrote ${OUT_CSV}"
echo "[sweep] wrote ${OUT_TEX}"
