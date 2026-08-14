#!/usr/bin/env bash
set -Eeuo pipefail

# Local MPA (Multi-Start Particle Ascent, --algorithm new_ppa) launcher.
#
# MPA approximates the regularized WDRO inner maximization by alternating,
# for R rounds per batch:
#   (a) K parallel BB+Armijo ascent steps on each particle z_i for the
#       objective  primary(clf(z_i), y_i) - lambda * c(z_i, x_i),
#       with the transport penalty ALWAYS anchored at the original x_i;
#   (b) within-class free-weight reassignment
#       z_i <- argmax_{z_j : y_j = y_i}  primary(clf(z_j), y_i) - lambda * c(z_j, x_i).
# The defaults here are the exact 2-round MPA schedule: 20 ascent steps ->
# reassign -> 20 ascent steps (from the reassigned particles, still anchored
# at the original x) -> final reassign. LOCAL_PPA_MIN_ROUNDS=2 guarantees the
# gain-based early stop can never skip the post-reassignment ascent burst.
#
# Round order (LOCAL_PPA_ROUND_ORDER):
#   * ascent_first (default; Algorithm 1) — rounds of {ascend; reassign};
#     R=1 is plain PA (no reassignment).
#   * reassign_first — mirrored rounds of {reassign; ascend} closed by a
#     FINAL reassignment (R=3: R-A-R-A-R-A-R): the round-0 reassignment
#     fires at z=x, so each sample's ascent STARTS from the best same-class
#     training sample under the DRO score, and — like ascent_first — the
#     theta-update always sees within-class best responses. All other knobs
#     (R, K, LRs, min_rounds, gain_rtol) are shared between the two orders.
#   * reassign_first_v2 — identical to reassign_first except the ROUND-0
#     reassignment is priced at lambda/LOCAL_PPA_ROUND0_LAMBDA_DAMPING
#     (graduated multi-start). At the full lambda=30 the round-0
#     reassignment at z=x is inert (~0.3% jumps on the 95.3% R2.pth); the
#     damped score activates confidence-driven jumps while the retained
#     cost term keeps assignments anchor-dependent (no particle collapse).
#     Rounds >= 1, the closing reassignment, and the ascent objective all
#     use the full lambda. Damping 1.0 == reassign_first exactly. The
#     damping is read ONLY by reassign_first_v2; ascent_first and
#     reassign_first ignore it.
#
# Step rule (LOCAL_PPA_STEP_RULE):
#   * const_lr (default here) — faithful MNIST/paper semantics: per-sample
#     updates, round 0 on the WRM diminishing schedule
#     ppa_round0_lr/sqrt(s), refinement rounds at the constant
#     ppa_refine_lr. Per-sample => the ascent is identical however the
#     batch is sharded across GPUs. LRs below were calibrated on the
#     CURRENT R2.pth (95.3%-clean PreActResNet18, 2026-07-08 swap; margin,
#     lambda=30 nmse): round0 0.03 -> inner obj ~46 at pixel-L2 ~0.59;
#     0.1 -> L2 1.28 / acc 0%. The old 87.3% R2_Old.pth wanted 0.1/0.05.
#   * bb_armijo — the repo-wide shared line-searched step rule (one step
#     size per batch-mean step), for parity with the other DRO adversaries.
#
# NOTES specific to new_ppa:
#   * MPA is stateless/transductive: warmup epochs would run the adversary
#     with frozen theta and learn nothing, so LOCAL_WARMUP_EPOCHS defaults 0.
#   * ATTACK_CLEAN_CORRECT_ONLY is honored (default 1): the MPA batch is
#     the clean-correct subset; misclassified samples stay at x and are
#     excluded from the reassignment pool.
#   * Under DDP the reassignment pool is ALL-GATHERED: each sample
#     best-responds over the full global batch (COMMON_BATCH candidates,
#     ~51 per class at 512) regardless of LOCAL_NPROC — 1, 2, or 4 GPUs run
#     the same MPA semantics.
#   * transport_for_eval runs a fresh MPA attack with predicted labels, so
#     the per-epoch transport/adv_acc eval columns measure robustness to
#     the MPA adversary (adds inner-loop cost to each epoch's eval);
#     input_pgd_acc remains the PGD benchmark number.
#
# Override any knob with an environment variable, e.g.
#   LOCAL_K=20 LOCAL_PPA_STEP_RULE=bb_armijo bash run_local_mpa.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCAL_VENV=${LOCAL_VENV:-/mloscratch/homes/aabdolla/optiselect/.venv}
source "${LOCAL_VENV}/bin/activate"

# ---------------------------------------------------------------------------
# Launch, paths, reproducibility
# ---------------------------------------------------------------------------
LOCAL_SEED=${LOCAL_SEED:-1}
LOCAL_NPROC=${LOCAL_NPROC:-1}
LOCAL_CUDA_VISIBLE_DEVICES=${LOCAL_CUDA_VISIBLE_DEVICES:-0}
LOCAL_SRC_DIR=${LOCAL_SRC_DIR:-${SCRIPT_DIR}}
LOCAL_PRETRAINED_PATH=${LOCAL_PRETRAINED_PATH:-${LOCAL_SRC_DIR}/ResNet_checkpoints/R2.pth}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-${LOCAL_SRC_DIR}/data}
LOCAL_RESULTS_ROOT=${LOCAL_RESULTS_ROOT:-${LOCAL_SRC_DIR}/input_icnn_ddp_runs}
LOCAL_NUM_WORKERS=${LOCAL_NUM_WORKERS:-2}
LOCAL_OMP_NUM_THREADS=${LOCAL_OMP_NUM_THREADS:-4}
LOCAL_MKL_NUM_THREADS=${LOCAL_MKL_NUM_THREADS:-4}
LOCAL_NCCL_DEBUG=${LOCAL_NCCL_DEBUG:-WARN}

# ---------------------------------------------------------------------------
# MPA inner loop (Algorithm 1)
# ---------------------------------------------------------------------------
LOCAL_K=${LOCAL_K:-20}                                # K ascent steps per round (round0 AND refine)
LOCAL_PPA_STEP_RULE=${LOCAL_PPA_STEP_RULE:-const_lr}  # const_lr (MNIST/paper) | bb_armijo (shared rule)
LOCAL_PPA_ROUND_ORDER=${LOCAL_PPA_ROUND_ORDER:-ascent_first}  # ascent_first | reassign_first | reassign_first_v2
LOCAL_PPA_ROUND0_LAMBDA_DAMPING=${LOCAL_PPA_ROUND0_LAMBDA_DAMPING:-6.0}  # reassign_first_v2 ONLY: round-0 priced at lambda/alpha
                                                      # (lam=30, alpha=6 -> lam0=5 -> ~53% round-0 jumps; alpha=10 -> ~73%)
LOCAL_PPA_NUM_ROUNDS=${LOCAL_PPA_NUM_ROUNDS:-2}       # R (2 = ascent, reassign, ascent, reassign)
LOCAL_PPA_MIN_ROUNDS=${LOCAL_PPA_MIN_ROUNDS:-2}       # >= NUM_ROUNDS disables the early stop entirely
LOCAL_PPA_ROUND0_STEPS=${LOCAL_PPA_ROUND0_STEPS:-}    # empty -> LOCAL_K
LOCAL_PPA_REFINE_STEPS=${LOCAL_PPA_REFINE_STEPS:-}    # empty -> LOCAL_K
LOCAL_PPA_ROUND0_LR=${LOCAL_PPA_ROUND0_LR:-0.03}      # const_lr: round0 base LR (diminishing /sqrt(s));
                                                      # calibrated for the 95.3% R2.pth (old model: 0.1)
LOCAL_PPA_REFINE_LR=${LOCAL_PPA_REFINE_LR:-0.015}     # const_lr: refinement constant LR (2:1 ratio)
LOCAL_PPA_GAIN_RTOL=${LOCAL_PPA_GAIN_RTOL:-1e-4}      # inert while MIN_ROUNDS >= NUM_ROUNDS

# ---------------------------------------------------------------------------
# Training schedule and outer classifier optimization (flagship recipe)
# ---------------------------------------------------------------------------
LOCAL_ADV_EPOCHS=${LOCAL_ADV_EPOCHS:-50}
LOCAL_WARMUP_EPOCHS=${LOCAL_WARMUP_EPOCHS:-0}         # no-op for stateless MPA; keep 0
LOCAL_COMMON_BATCH=${LOCAL_COMMON_BATCH:-512}         # GLOBAL batch, split across ranks
LOCAL_LR_THETA=${LOCAL_LR_THETA:-0.05}
LOCAL_PENALTY_LAMBDA=${LOCAL_PENALTY_LAMBDA:-30}
LOCAL_LAMBDA_SCHEDULE=${LOCAL_LAMBDA_SCHEDULE:-}
LOCAL_LAMBDA_STAGE_EPOCHS=${LOCAL_LAMBDA_STAGE_EPOCHS:-0}
LOCAL_TRANSPORT_COST=${LOCAL_TRANSPORT_COST:-normalized_mse}  # legacy lambda=30 convention
LOCAL_USE_MARGIN_LOSS=${LOCAL_USE_MARGIN_LOSS:-1}     # 1=margin inner objective (repo flagship);
                                                      # 0=CE makes inner f identical to the outer CE
LOCAL_ATTACK_CLEAN_CORRECT_ONLY=${LOCAL_ATTACK_CLEAN_CORRECT_ONLY:-1}
LOCAL_FREEZE_BATCHNORM=${LOCAL_FREEZE_BATCHNORM:-0}
LOCAL_FREEZE_BATCHNORM_AFFINE=${LOCAL_FREEZE_BATCHNORM_AFFINE:-0}
LOCAL_ADVERSARY_CLASSIFIER_EVAL=${LOCAL_ADVERSARY_CLASSIFIER_EVAL:-1}

# ---------------------------------------------------------------------------
# Shared BB+Armijo step rule (same defaults as every other DRO adversary)
# ---------------------------------------------------------------------------
LOCAL_BB_ALPHA0=${LOCAL_BB_ALPHA0:-5e-4}
LOCAL_BB_ALPHA_MIN=${LOCAL_BB_ALPHA_MIN:-1e-6}
LOCAL_BB_ALPHA_MAX=${LOCAL_BB_ALPHA_MAX:-1.0}
LOCAL_BB_LS_C=${LOCAL_BB_LS_C:-0.1}
LOCAL_BB_LS_SHRINK=${LOCAL_BB_LS_SHRINK:-0.5}
LOCAL_BB_LS_MAX_STEPS=${LOCAL_BB_LS_MAX_STEPS:-10}

# ---------------------------------------------------------------------------
# Input-PGD evaluation (legacy pixel-L2 eps=0.5 benchmark)
# ---------------------------------------------------------------------------
LOCAL_EVAL_PGD_SAMPLES=${LOCAL_EVAL_PGD_SAMPLES:-1024}
# MPA's transport eval is a full inner maximization per batch; cap it like
# the PGD eval so rank-0 eval time stays bounded (0 = full 10k test set).
LOCAL_EVAL_TRANSPORT_SAMPLES=${LOCAL_EVAL_TRANSPORT_SAMPLES:-1024}
LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT=${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT:-0}  # identity transport -> alignment meaningless
LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES=${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES:-256}
LOCAL_INP_P=${LOCAL_INP_P:-2}
LOCAL_INP_EPS=${LOCAL_INP_EPS:-0.5}
LOCAL_INP_STEPS=${LOCAL_INP_STEPS:-20}
LOCAL_INP_RESTARTS=${LOCAL_INP_RESTARTS:-5}
LOCAL_INPUT_PGD_LOSS=${LOCAL_INPUT_PGD_LOSS:-ce}
LOCAL_INPUT_PGD_GEOMETRY=${LOCAL_INPUT_PGD_GEOMETRY:-pixel_l2_squared}
LOCAL_SKIP_PGD_DURING_TRAIN=${LOCAL_SKIP_PGD_DURING_TRAIN:-0}

# ---------------------------------------------------------------------------
# Checkpointing / misc
# ---------------------------------------------------------------------------
LOCAL_BENCHMARK_MODE=${LOCAL_BENCHMARK_MODE:-0}
LOCAL_PROFILE_INNER=${LOCAL_PROFILE_INNER:-0}
LOCAL_PROFILE_INNER_BATCHES=${LOCAL_PROFILE_INNER_BATCHES:-0}
LOCAL_RESUME_CHECKPOINT=${LOCAL_RESUME_CHECKPOINT:-}
LOCAL_SAVE_EVERY_EPOCH_DIR=${LOCAL_SAVE_EVERY_EPOCH_DIR:-}
LOCAL_CHECKPOINT_EPOCH_OFFSET=${LOCAL_CHECKPOINT_EPOCH_OFFSET:-0}

tag_token() {
  printf '%s' "$1" | tr -s '[:space:]' 'x' | sed -e 's/\./p/g' -e 's/-/m/g' -e 's/,/_/g'
}

if [[ "${LOCAL_USE_MARGIN_LOSS}" =~ ^(1|true|True|TRUE|yes|Yes|YES|on|On|ON)$ ]]; then
  TRAIN_ADV_LOSS_TAG="margin"
  LOCAL_USE_MARGIN_LOSS=1
else
  TRAIN_ADV_LOSS_TAG="ce"
  LOCAL_USE_MARGIN_LOSS=0
fi

R0_TAG="$(tag_token "${LOCAL_PPA_ROUND0_STEPS:-${LOCAL_K}}")"
RF_TAG="$(tag_token "${LOCAL_PPA_REFINE_STEPS:-${LOCAL_K}}")"
RULE_TAG="$(tag_token "${LOCAL_PPA_STEP_RULE}")"
if [ "${LOCAL_PPA_STEP_RULE}" = "const_lr" ]; then
  RULE_TAG="${RULE_TAG}$(tag_token "${LOCAL_PPA_ROUND0_LR}")x$(tag_token "${LOCAL_PPA_REFINE_LR}")"
fi
MASK_TAG="ccm$(tag_token "${LOCAL_ATTACK_CLEAN_CORRECT_ONLY}")"
ORDER_TAG="af"
if [ "${LOCAL_PPA_ROUND_ORDER}" = "reassign_first" ]; then
  ORDER_TAG="rf"
elif [ "${LOCAL_PPA_ROUND_ORDER}" = "reassign_first_v2" ]; then
  ORDER_TAG="rf2d$(tag_token "${LOCAL_PPA_ROUND0_LAMBDA_DAMPING}")"
fi
RUN_TAG=${RUN_TAG:-mpa_lam$(tag_token "${LOCAL_PENALTY_LAMBDA}")_cost$(tag_token "${LOCAL_TRANSPORT_COST}")_adv${TRAIN_ADV_LOSS_TAG}_R$(tag_token "${LOCAL_PPA_NUM_ROUNDS}")${ORDER_TAG}_K${R0_TAG}x${RF_TAG}_${RULE_TAG}_${MASK_TAG}_lr$(tag_token "${LOCAL_LR_THETA}")_ep${LOCAL_ADV_EPOCHS}_bs${LOCAL_COMMON_BATCH}_ddp${LOCAL_NPROC}_seed${LOCAL_SEED}}

export OMP_NUM_THREADS="${LOCAL_OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${LOCAL_MKL_NUM_THREADS}"
export NCCL_DEBUG="${LOCAL_NCCL_DEBUG}"

CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" \
SRC_DIR="${LOCAL_SRC_DIR}" \
PRETRAINED_PATH="${LOCAL_PRETRAINED_PATH}" \
DATA_DIR="${LOCAL_DATA_DIR}" \
RESULTS_ROOT="${LOCAL_RESULTS_ROOT}" \
RUN_NAME="${RUN_TAG}" \
OUTPUT_FOLDER_NAME="${RUN_TAG}" \
NUM_WORKERS="${LOCAL_NUM_WORKERS}" \
COMMON_BATCH="${LOCAL_COMMON_BATCH}" \
LR_THETA="${LOCAL_LR_THETA}" \
PENALTY_LAMBDA="${LOCAL_PENALTY_LAMBDA}" \
LAMBDA_SCHEDULE="${LOCAL_LAMBDA_SCHEDULE}" \
LAMBDA_STAGE_EPOCHS="${LOCAL_LAMBDA_STAGE_EPOCHS}" \
TRANSPORT_COST="${LOCAL_TRANSPORT_COST}" \
USE_MARGIN_LOSS="${LOCAL_USE_MARGIN_LOSS}" \
ATTACK_CLEAN_CORRECT_ONLY="${LOCAL_ATTACK_CLEAN_CORRECT_ONLY}" \
EPOCHS_ICNN_PRETRAIN="${LOCAL_WARMUP_EPOCHS}" \
FREEZE_BATCHNORM="${LOCAL_FREEZE_BATCHNORM}" \
FREEZE_BATCHNORM_AFFINE="${LOCAL_FREEZE_BATCHNORM_AFFINE}" \
ADVERSARY_CLASSIFIER_EVAL="${LOCAL_ADVERSARY_CLASSIFIER_EVAL}" \
PPA_STEP_RULE="${LOCAL_PPA_STEP_RULE}" \
PPA_ROUND_ORDER="${LOCAL_PPA_ROUND_ORDER}" \
PPA_ROUND0_LAMBDA_DAMPING="${LOCAL_PPA_ROUND0_LAMBDA_DAMPING}" \
PPA_NUM_ROUNDS="${LOCAL_PPA_NUM_ROUNDS}" \
PPA_MIN_ROUNDS="${LOCAL_PPA_MIN_ROUNDS}" \
PPA_ROUND0_STEPS="${LOCAL_PPA_ROUND0_STEPS}" \
PPA_REFINE_STEPS="${LOCAL_PPA_REFINE_STEPS}" \
PPA_ROUND0_LR="${LOCAL_PPA_ROUND0_LR}" \
PPA_REFINE_LR="${LOCAL_PPA_REFINE_LR}" \
PPA_GAIN_RTOL="${LOCAL_PPA_GAIN_RTOL}" \
BB_ALPHA0="${LOCAL_BB_ALPHA0}" \
BB_ALPHA_MIN="${LOCAL_BB_ALPHA_MIN}" \
BB_ALPHA_MAX="${LOCAL_BB_ALPHA_MAX}" \
BB_LS_C="${LOCAL_BB_LS_C}" \
BB_LS_SHRINK="${LOCAL_BB_LS_SHRINK}" \
BB_LS_MAX_STEPS="${LOCAL_BB_LS_MAX_STEPS}" \
INP_P="${LOCAL_INP_P}" \
INP_EPS="${LOCAL_INP_EPS}" \
INP_STEPS="${LOCAL_INP_STEPS}" \
INP_RESTARTS="${LOCAL_INP_RESTARTS}" \
EVAL_PGD_SAMPLES="${LOCAL_EVAL_PGD_SAMPLES}" \
EVAL_TRANSPORT_SAMPLES="${LOCAL_EVAL_TRANSPORT_SAMPLES}" \
EVAL_TRANSPORT_PGD_ALIGNMENT="${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT}" \
EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES="${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES}" \
INPUT_PGD_LOSS="${LOCAL_INPUT_PGD_LOSS}" \
INPUT_PGD_GEOMETRY="${LOCAL_INPUT_PGD_GEOMETRY}" \
SKIP_PGD_DURING_TRAIN="${LOCAL_SKIP_PGD_DURING_TRAIN}" \
BENCHMARK_MODE="${LOCAL_BENCHMARK_MODE}" \
PROFILE_INNER="${LOCAL_PROFILE_INNER}" \
PROFILE_INNER_BATCHES="${LOCAL_PROFILE_INNER_BATCHES}" \
RESUME_CHECKPOINT="${LOCAL_RESUME_CHECKPOINT}" \
SAVE_EVERY_EPOCH_DIR="${LOCAL_SAVE_EVERY_EPOCH_DIR}" \
CHECKPOINT_EPOCH_OFFSET="${LOCAL_CHECKPOINT_EPOCH_OFFSET}" \
OMEGA_STEPS="${LOCAL_K}" \
RUN_ONLY_ALGO=new_ppa \
bash run_runtime_sweep_ddp.sh \
  0 \
  "${LOCAL_SEED}" \
  "${LOCAL_NPROC}" \
  "${LOCAL_K}" \
  "${LOCAL_ADV_EPOCHS}" \
  "${RUN_TAG}"
