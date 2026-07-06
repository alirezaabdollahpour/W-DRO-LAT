#!/usr/bin/env bash
set -Eeuo pipefail

# Local NPF / NPF-LastQuad launcher.
#
# This file is intentionally explicit: every knob below is either forwarded to
# run_runtime_sweep_ddp.sh or controls the local launch wrapper. Override any
# setting with an environment variable, e.g.
#
#   LOCAL_K=10 LOCAL_INPUT_PGD_LOSS=margin bash run_local_npf_lastquad.sh
#
# Defaults are the measured-best NPF-LastQuad recipe (2026-07-06, second
# audit, workflow wf_94b9b822 + training run ablC_lastquad_rank64_lam30nmse):
# logsumexp-margin adversary loss, PERSISTENT omega with 2 adversary-only
# warmup epochs, principled log-normal random init, exp-reparametrized
# nonnegative hidden weights AND exp-reparametrized (never-dead) PosDef
# diagonals, OUTPUT RANK 64 (the decisive knob: rank 2 confines the ascent to
# 1-2 shared far-field directions that the classifier neutralizes in one
# step; rank 64 lifted PGD@0.5 from ~10% to 45%+), literal lambda=30 on the
# legacy normalized-MSE cost, omega weight decay 1e-4 (legacy parity; taxes
# far-field kernel growth), LR_theta=0.05 over 50 adv epochs, and evaluation
# at raw pixel-L2 eps=0.5 (CE loss, 20 steps, 5 restarts, 1024 samples).
# NOTE: the eval here is STRICTER than the legacy 57% benchmark (clean-correct
# AND adv-correct numerator + best-iterate tracking vs legacy's endpoint-only
# adv-correct/total), so equal true robustness reads a few points lower.
# The first 2026-07-06 audit (wf_fb1bbd0e) fixed the degenerate inner
# maximizer (CE + per-batch omega reset + identity init + lambda 0.0976 +
# eval at rho=0.05 ~ eps 1.45 = classifier trained on clean data).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCAL_VENV=${LOCAL_VENV:-/mloscratch/homes/aabdolla/optiselect/.venv}
source "${LOCAL_VENV}/bin/activate"

# ---------------------------------------------------------------------------
# Launch, paths, and reproducibility
# ---------------------------------------------------------------------------
LOCAL_SPLIT=${LOCAL_SPLIT:-0}                         # 0/1 run selected NPF algo
LOCAL_RUN_ONLY_ALGO=${LOCAL_RUN_ONLY_ALGO:-npf_lastquad} # npf_lastquad | npf
LOCAL_SEED=${LOCAL_SEED:-1}
LOCAL_NPROC=${LOCAL_NPROC:-1}                         # DDP world size
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
# Training schedule and outer classifier optimization
# ---------------------------------------------------------------------------
# Per microbatch protocol:
#   1. run LOCAL_K NPF omega-ascent steps on the current microbatch,
#   2. transport that same microbatch with the updated NPF map,
#   3. run one classifier/theta update on those transported samples.
# Warmup epochs do steps 1/2 only and skip step 3. The legacy golden run used
# 2 warmup epochs + 30 adversarial epochs with a PERSISTENT omega; the
# default here extends to 50 adversarial epochs (cosine T_max tracks it
# automatically) with LR_theta=0.05.
LOCAL_K=${LOCAL_K:-10}
LOCAL_ADV_EPOCHS=${LOCAL_ADV_EPOCHS:-50}
LOCAL_WARMUP_EPOCHS=${LOCAL_WARMUP_EPOCHS:-2}
LOCAL_COMMON_BATCH=${LOCAL_COMMON_BATCH:-512}         # global batch, split across ranks (legacy golden used 512)
# 0.05: half the legacy 0.1. At 0.1 the classifier races ahead of the
# adversary early (ablA's PGD peaked at ep6 then decayed); 0.05 + 50-epoch
# cosine gives the persistent omega more epochs of pressure per LR level.
LOCAL_LR_THETA=${LOCAL_LR_THETA:-0.05}

# DRO objective: adversary maximizes loss - lambda * transport_cost.
# Literal legacy semantics: lambda=30 on the normalized-MSE cost (per-sample
# mean squared difference over the 3072 normalized coordinates). This is the
# exact golden-run convention and what the winning ablC run used.
# Pixel-space equivalent: LOCAL_TRANSPORT_COST=pixel_l2_squared
# LOCAL_PENALTY_LAMBDA=0.242 (= 30 / (3072 * mean(std^2)),
# std=(0.2023,0.1994,0.2010)).
LOCAL_PENALTY_LAMBDA=${LOCAL_PENALTY_LAMBDA:-30}
LOCAL_LAMBDA_SCHEDULE=${LOCAL_LAMBDA_SCHEDULE:-}      # e.g. "30,20,15,10,5"
LOCAL_LAMBDA_STAGE_EPOCHS=${LOCAL_LAMBDA_STAGE_EPOCHS:-0}
LOCAL_TRANSPORT_COST=${LOCAL_TRANSPORT_COST:-normalized_mse}  # normalized_mse (legacy) | pixel_l2_squared
# The legacy 57% run used the logsumexp-margin adversary loss. CE on the
# clean-correct subset of a confident pretrained classifier is saturated
# (CE ~0.07, omega grad ~0.02) and deadlocks the BB+Armijo inner ascent.
LOCAL_USE_MARGIN_LOSS=${LOCAL_USE_MARGIN_LOSS:-1}     # 0=CE, 1=margin
LOCAL_ATTACK_CLEAN_CORRECT_ONLY=${LOCAL_ATTACK_CLEAN_CORRECT_ONLY:-1}

# ---------------------------------------------------------------------------
# BatchNorm policy
# ---------------------------------------------------------------------------
# 1 = attack the same eval-mode classifier used by PGD evaluation (aligned
# with the threat model). This is what the winning rank-64 run (ablC, 45%+
# PGD@0.5) used, so it stays the default. NOTE: the legacy 57% golden run
# attacked the TRAIN-mode classifier (batch-stat BN) during minimax epochs —
# batch-stat BN structurally cancels batch-constant translations, one more
# force against far-field cheats; set LOCAL_ADVERSARY_CLASSIFIER_EVAL=0 for
# exact legacy parity (supported and worth an ablation arm). Warmup always
# uses eval mode, which IS legacy parity (legacy warmup ran head.train(False)).
# BatchNorm can still participate in the outer classifier update unless
# LOCAL_FREEZE_BATCHNORM=1.
LOCAL_FREEZE_BATCHNORM=${LOCAL_FREEZE_BATCHNORM:-0}
LOCAL_FREEZE_BATCHNORM_AFFINE=${LOCAL_FREEZE_BATCHNORM_AFFINE:-0}
LOCAL_ADVERSARY_CLASSIFIER_EVAL=${LOCAL_ADVERSARY_CLASSIFIER_EVAL:-1}
LOCAL_BATCHNORM_ONLINE_REFRESH=${LOCAL_BATCHNORM_ONLINE_REFRESH:-0}
LOCAL_BATCHNORM_ONLINE_REFRESH_MOMENTUM=${LOCAL_BATCHNORM_ONLINE_REFRESH_MOMENTUM:-}
LOCAL_RECALIBRATE_BATCHNORM=${LOCAL_RECALIBRATE_BATCHNORM:-0}
LOCAL_BATCHNORM_RECALIBRATION_BATCHES=${LOCAL_BATCHNORM_RECALIBRATION_BATCHES:-0}
LOCAL_BATCHNORM_RECALIBRATION_RESET=${LOCAL_BATCHNORM_RECALIBRATION_RESET:-0}
LOCAL_BATCHNORM_RECALIBRATION_MOMENTUM=${LOCAL_BATCHNORM_RECALIBRATION_MOMENTUM:-}

# ---------------------------------------------------------------------------
# NPF-LastQuad architecture
# ---------------------------------------------------------------------------
# Faithful OTT-style pieces:
#   * activation is applied through the scalar residual layer;
#   * final identity positive-definite potential is always present;
#   * LastQuad keeps hidden input skips affine and learns the final residual
#     input quadratic only.
LOCAL_NPF_LASTQUAD_HIDDEN=${LOCAL_NPF_LASTQUAD_HIDDEN:-"1024 512 512 256 128 64"}  # legacy golden-run widths
LOCAL_NPF_LASTQUAD_ACTIVATION=${LOCAL_NPF_LASTQUAD_ACTIVATION:-softplus}  # relu | softplus | elu
LOCAL_NPF_LASTQUAD_ELU_ALPHA=${LOCAL_NPF_LASTQUAD_ELU_ALPHA:-1.0}
LOCAL_NPF_LASTQUAD_SOFTPLUS_BETA=${LOCAL_NPF_LASTQUAD_SOFTPLUS_BETA:-20.0}
# Output rank 64 is the decisive capacity knob (2026-07-06 probe + ablC run):
# at rank 2, >50% of the omega gradient funnels into 1-2 shared far-field
# directions (constant translation / global rescale) that the classifier
# neutralizes in one SGD step and PGD@0.5 plateaus at 5-16%; at rank 64 the
# same K=10 BB+Armijo ascent produces sample-dependent transports
# (top-1 SVD energy 26% vs 46%) and full training reached 45%+ and climbing.
LOCAL_NPF_LASTQUAD_OUTPUT_RANK=${LOCAL_NPF_LASTQUAD_OUTPUT_RANK:-64}
LOCAL_NPF_LASTQUAD_STRONG_CONVEXITY=${LOCAL_NPF_LASTQUAD_STRONG_CONVEXITY:-1.0}
# identity_init=0 keeps the constructor's principled log-normal RANDOM init
# (the legacy golden run started random). The audit showed that at identity
# init 99.9% of the omega gradient energy funnels into lin_kernel, whose
# transport effect is a sample-INDEPENDENT constant translation — the
# mechanism behind "large transport norm but orthogonal to PGD". Keep
# identity init only for diagnostics.
LOCAL_NPF_LASTQUAD_IDENTITY_INIT=${LOCAL_NPF_LASTQUAD_IDENTITY_INIT:-0}
LOCAL_NPF_LASTQUAD_INIT_EPS=${LOCAL_NPF_LASTQUAD_INIT_EPS:-1e-2}
LOCAL_NPF_LASTQUAD_POS_WEIGHTS=${LOCAL_NPF_LASTQUAD_POS_WEIGHTS:-1}
# exp = exact legacy NonNegativeLinear reparametrization: strictly positive
# weights, never-dead gradients. relu permanently kills weights whose raw
# value goes negative; softplus freezes them (sigmoid(raw) ~ 0 near zero).
LOCAL_NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER=${LOCAL_NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER:-exp}

# ---------------------------------------------------------------------------
# Full NPF architecture
# ---------------------------------------------------------------------------
# Use LOCAL_RUN_ONLY_ALGO=npf for the wider OTT NPF ablation. This is the
# important follow-up if LastQuad remains nearly orthogonal to PGD.
LOCAL_NPF_HIDDEN=${LOCAL_NPF_HIDDEN:-"1024 512 512 256 128 64"}  # legacy golden-run widths
LOCAL_NPF_OUTER_RANK=${LOCAL_NPF_OUTER_RANK:-8}
LOCAL_NPF_INNER_RANK=${LOCAL_NPF_INNER_RANK:-2}
LOCAL_NPF_ACTIVATION=${LOCAL_NPF_ACTIVATION:-softplus}  # relu | softplus | elu
LOCAL_NPF_ELU_ALPHA=${LOCAL_NPF_ELU_ALPHA:-1.0}
LOCAL_NPF_SOFTPLUS_BETA=${LOCAL_NPF_SOFTPLUS_BETA:-20.0}
LOCAL_NPF_STRONG_CONVEXITY=${LOCAL_NPF_STRONG_CONVEXITY:-1.0}
# Identity-adjacent start: the 2026-07-06 probe measured the non-identity
# all-layers init 12.9 pixel-L2 from identity with inner objective -40 (the
# per-layer lecun lin/quad kernels add up); K=10/batch then only shrinks its
# own residual. identity+eps=1e-2 is the trainable near-identity start.
LOCAL_NPF_IDENTITY_INIT=${LOCAL_NPF_IDENTITY_INIT:-1}
LOCAL_NPF_INIT_EPS=${LOCAL_NPF_INIT_EPS:-1e-2}
LOCAL_NPF_POS_WEIGHTS=${LOCAL_NPF_POS_WEIGHTS:-1}
LOCAL_NPF_POSITIVE_WEIGHT_RECTIFIER=${LOCAL_NPF_POSITIVE_WEIGHT_RECTIFIER:-exp}  # exp = legacy-exact, never-dead

# ---------------------------------------------------------------------------
# Inner optimizer for the NPF adversary
# ---------------------------------------------------------------------------
LOCAL_NPF_INNER_OPTIMIZER=${LOCAL_NPF_INNER_OPTIMIZER:-bb_armijo} # bb_armijo | muon
# 0 = persistent omega across batches/epochs (legacy semantics; ~31k cumulative
# ascent steps over a full run). 1 resets to the init snapshot every batch,
# which gives the adversary exactly K cold-start steps and never accumulates
# anything — with K=10 the classifier effectively trains on clean data.
LOCAL_NPF_RESET_OMEGA_EACH_BATCH=${LOCAL_NPF_RESET_OMEGA_EACH_BATCH:-0}
# For PGD-targeted robust accuracy, train on NPF transports clipped to the same
# input threat set used by evaluation. Set to 0 for pure unconstrained WDRO.
LOCAL_NPF_PROJECT_TO_INPUT_BALL=${LOCAL_NPF_PROJECT_TO_INPUT_BALL:-0}
# Optional PGD anchor for the NPF inner objective. This directly addresses the
# observed failure mode where the learned transport has large norm but points
# almost orthogonally to PGD. Set weight=0 for pure NPF/LastQuad ablations.
LOCAL_NPF_PGD_ANCHOR_WEIGHT=${LOCAL_NPF_PGD_ANCHOR_WEIGHT:-0}
LOCAL_NPF_PGD_ANCHOR_STEPS=${LOCAL_NPF_PGD_ANCHOR_STEPS:-0}
LOCAL_NPF_PGD_ANCHOR_RESTARTS=${LOCAL_NPF_PGD_ANCHOR_RESTARTS:-0}
LOCAL_NPF_PGD_ANCHOR_LOSS=${LOCAL_NPF_PGD_ANCHOR_LOSS:-margin}
LOCAL_RESET_PARAMETRIC_BB_EACH_BATCH=${LOCAL_RESET_PARAMETRIC_BB_EACH_BATCH:-1}
LOCAL_PARAMETRIC_BB_MAX_GRAD_NORM=${LOCAL_PARAMETRIC_BB_MAX_GRAD_NORM:-1.0}
# L2 decay on the FREE omega kernels (affine input skips, PosDef lin/quad)
# inside the ascent objective. Legacy golden-run parity: 1e-4. This is the
# slow force that taxes unbounded growth of the sample-independent
# translation/rescale kernels across ~30k persistent-omega ascent steps.
LOCAL_NPF_OMEGA_WEIGHT_DECAY=${LOCAL_NPF_OMEGA_WEIGHT_DECAY:-1e-4}

# BB+Armijo defaults match the stable ICNN launcher.
LOCAL_BB_ALPHA0=${LOCAL_BB_ALPHA0:-5e-4}
LOCAL_BB_ALPHA_MIN=${LOCAL_BB_ALPHA_MIN:-1e-6}
LOCAL_BB_ALPHA_MAX=${LOCAL_BB_ALPHA_MAX:-1.0}
LOCAL_BB_LS_C=${LOCAL_BB_LS_C:-0.1}
LOCAL_BB_LS_SHRINK=${LOCAL_BB_LS_SHRINK:-0.5}
LOCAL_BB_LS_MAX_STEPS=${LOCAL_BB_LS_MAX_STEPS:-10}

# Muon is available for NPF omega parameters; leave these explicit for ablations.
LOCAL_NPF_MUON_LR=${LOCAL_NPF_MUON_LR:-2e-4}
LOCAL_NPF_MUON_MOMENTUM=${LOCAL_NPF_MUON_MOMENTUM:-0.95}
LOCAL_NPF_MUON_NESTEROV=${LOCAL_NPF_MUON_NESTEROV:-1}
LOCAL_NPF_MUON_NS_STEPS=${LOCAL_NPF_MUON_NS_STEPS:-5}
LOCAL_NPF_MUON_MATRIX_LR_SCALE=${LOCAL_NPF_MUON_MATRIX_LR_SCALE:-auto}
LOCAL_NPF_MUON_WEIGHT_DECAY=${LOCAL_NPF_MUON_WEIGHT_DECAY:-0.0}
LOCAL_NPF_MUON_FALLBACK=${LOCAL_NPF_MUON_FALLBACK:-adamw}
LOCAL_NPF_MUON_FALLBACK_LR=${LOCAL_NPF_MUON_FALLBACK_LR:-0.0}
LOCAL_NPF_MUON_FALLBACK_WEIGHT_DECAY=${LOCAL_NPF_MUON_FALLBACK_WEIGHT_DECAY:-0.0}
LOCAL_NPF_MUON_ADAM_BETA1=${LOCAL_NPF_MUON_ADAM_BETA1:-0.9}
LOCAL_NPF_MUON_ADAM_BETA2=${LOCAL_NPF_MUON_ADAM_BETA2:-0.999}
LOCAL_NPF_MUON_ADAM_EPS=${LOCAL_NPF_MUON_ADAM_EPS:-1e-8}
LOCAL_NPF_MUON_MAX_GRAD_NORM=${LOCAL_NPF_MUON_MAX_GRAD_NORM:-0.0}

# ---------------------------------------------------------------------------
# Input-PGD evaluation
# ---------------------------------------------------------------------------
# WRM-style PGM/PGD attacks maximize the model loss; CE is the faithful
# default. Set LOCAL_INPUT_PGD_LOSS=margin for the stronger local evaluator.
LOCAL_EVAL_PGD_SAMPLES=${LOCAL_EVAL_PGD_SAMPLES:-1024}  # legacy benchmark used ceil(1000/512)*512
LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT=${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT:-1}
LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES=${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES:-256}
LOCAL_INP_P=${LOCAL_INP_P:-2}
# WRM paper / GFS-DRO protocol: attack plots report
# rho = epsilon_adv / C_p, with C_p = E_train ||X||_p; when enabled the
# launcher interprets LOCAL_INP_EPS per LOCAL_WRM_PGD_EPS_MODE and converts.
# Default OFF: LOCAL_INP_EPS=0.5 is used directly as the raw pixel-L2 PGD
# radius (the legacy 57% benchmark radius). The protocol also requires the
# pixel transport cost, which is no longer the default.
LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL=${LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL:-0}
# raw + eps=0.5 is the legacy 57% benchmark radius (rho ~ 0.0173). The old
# default (normalized_radius, rho=0.05 -> eps ~ 1.447) is 2.9x larger and
# saturates the metric: the undefended classifier already reads ~0% there.
LOCAL_WRM_PGD_EPS_MODE=${LOCAL_WRM_PGD_EPS_MODE:-raw} # normalized_radius | raw
LOCAL_WRM_CP=${LOCAL_WRM_CP:-}                       # optional manual C_p override
LOCAL_WRM_CP_BATCH_SIZE=${LOCAL_WRM_CP_BATCH_SIZE:-1024}
LOCAL_WRM_CP_NUM_WORKERS=${LOCAL_WRM_CP_NUM_WORKERS:-0}
LOCAL_WRM_MAX_NORMALIZED_RADIUS=${LOCAL_WRM_MAX_NORMALIZED_RADIUS:-0.25}
LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS=${LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS:-0}
LOCAL_INP_EPS=${LOCAL_INP_EPS:-0.5}
LOCAL_INP_STEPS=${LOCAL_INP_STEPS:-20}
LOCAL_INP_RESTARTS=${LOCAL_INP_RESTARTS:-5}
LOCAL_INPUT_PGD_LOSS=${LOCAL_INPUT_PGD_LOSS:-ce}      # ce | margin
LOCAL_INPUT_PGD_GEOMETRY=${LOCAL_INPUT_PGD_GEOMETRY:-pixel_l2_squared}
LOCAL_SKIP_PGD_DURING_TRAIN=${LOCAL_SKIP_PGD_DURING_TRAIN:-0}

# ---------------------------------------------------------------------------
# Checkpointing, profiling, post phases
# ---------------------------------------------------------------------------
LOCAL_BENCHMARK_MODE=${LOCAL_BENCHMARK_MODE:-0}
LOCAL_PROFILE_INNER=${LOCAL_PROFILE_INNER:-0}
LOCAL_PROFILE_INNER_BATCHES=${LOCAL_PROFILE_INNER_BATCHES:-0}
LOCAL_RESUME_CHECKPOINT=${LOCAL_RESUME_CHECKPOINT:-}
LOCAL_SAVE_EVERY_EPOCH_DIR=${LOCAL_SAVE_EVERY_EPOCH_DIR:-}
LOCAL_CHECKPOINT_EPOCH_OFFSET=${LOCAL_CHECKPOINT_EPOCH_OFFSET:-0}
LOCAL_FROZEN_ADVERSARY_EPOCHS=${LOCAL_FROZEN_ADVERSARY_EPOCHS:-0}
LOCAL_FROZEN_ADVERSARY_MAP_STEPS=${LOCAL_FROZEN_ADVERSARY_MAP_STEPS:-1}

tag_token() {
  printf '%s' "$1" | tr -s '[:space:]' 'x' | sed -e 's/\./p/g' -e 's/-/m/g' -e 's/,/_/g'
}

is_truthy() {
  [[ "$1" =~ ^(1|true|True|TRUE|yes|Yes|YES|on|On|ON)$ ]]
}

is_falsy() {
  [[ "$1" =~ ^(0|false|False|FALSE|no|No|NO|off|Off|OFF)$ ]]
}

canonical_p_norm() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    2|2.0) printf '2' ;;
    inf|infinity) printf 'inf' ;;
    *)
      echo "[FATAL] WRM normalized-radius protocol supports LOCAL_INP_P=2 or inf, got: $1" >&2
      exit 1
      ;;
  esac
}

compute_wrm_cp() {
  local p_norm="$1"
  WRM_CP_P="${p_norm}" \
  WRM_CP_DATA_DIR="${LOCAL_DATA_DIR}" \
  WRM_CP_BATCH_SIZE="${LOCAL_WRM_CP_BATCH_SIZE}" \
  WRM_CP_NUM_WORKERS="${LOCAL_WRM_CP_NUM_WORKERS}" \
  python - <<'PY'
import os
import sys

from pretrained_input_icnn.data import get_cifar10_loaders
from pretrained_input_icnn.utils.transforms import to_pixel

p_norm = os.environ["WRM_CP_P"]
data_dir = os.environ["WRM_CP_DATA_DIR"]
batch_size = int(os.environ["WRM_CP_BATCH_SIZE"])
num_workers = int(os.environ["WRM_CP_NUM_WORKERS"])

train_loader, _, _ = get_cifar10_loaders(
    batch_size=batch_size,
    num_workers=num_workers,
    data_root=data_dir,
    augment_train=False,
    pin_memory=False,
    download=True,
    seed=0,
)

total = 0
norm_sum = 0.0
for x_norm, _ in train_loader:
    x_pix = to_pixel(x_norm).clamp(0.0, 1.0)
    flat = x_pix.flatten(1)
    if p_norm == "2":
        norms = flat.norm(p=2, dim=1)
    elif p_norm == "inf":
        norms = flat.abs().max(dim=1).values
    else:
        raise RuntimeError(f"unsupported p_norm={p_norm!r}")
    norm_sum += float(norms.sum().item())
    total += int(x_pix.size(0))

if total <= 0:
    print("C_p computation saw no CIFAR-10 training samples.", file=sys.stderr)
    sys.exit(1)

print(f"{norm_sum / total:.12g}")
PY
}

case "${LOCAL_RUN_ONLY_ALGO}" in
  npf_lastquad|npf) ;;
  *)
    echo "[FATAL] LOCAL_RUN_ONLY_ALGO must be npf_lastquad or npf, got: ${LOCAL_RUN_ONLY_ALGO}" >&2
    exit 1
    ;;
esac

if ! is_truthy "${LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL}" && ! is_falsy "${LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL}"; then
  echo "[FATAL] LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL must be true/false-like, got: ${LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL}" >&2
  exit 1
fi
if ! is_truthy "${LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS}" && ! is_falsy "${LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS}"; then
  echo "[FATAL] LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS must be true/false-like, got: ${LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS}" >&2
  exit 1
fi

if is_truthy "${LOCAL_USE_MARGIN_LOSS}"; then
  TRAIN_ADV_LOSS_TAG="margin"
  # run_runtime_sweep_ddp.sh tests USE_MARGIN_LOSS = "1" exactly; canonicalize
  # so "true"/"yes" cannot silently fall through to CE while the run name
  # claims margin.
  LOCAL_USE_MARGIN_LOSS=1
elif is_falsy "${LOCAL_USE_MARGIN_LOSS}"; then
  TRAIN_ADV_LOSS_TAG="ce"
  LOCAL_USE_MARGIN_LOSS=0
else
  echo "[FATAL] LOCAL_USE_MARGIN_LOSS must be true/false-like, got: ${LOCAL_USE_MARGIN_LOSS}" >&2
  exit 1
fi

LOCAL_INP_EPS_REQUESTED="${LOCAL_INP_EPS}"
LOCAL_WRM_CP_VALUE=""
LOCAL_WRM_NORMALIZED_RADIUS_VALUE=""
LOCAL_INP_EPS_PROTOCOL_TAG="raw"
if is_truthy "${LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL}"; then
  case "$(printf '%s' "${LOCAL_TRANSPORT_COST}" | tr '[:upper:]' '[:lower:]')" in
    pixel|pixel_l2|pixel_l2_squared) ;;
    *)
      echo "[FATAL] WRM normalized-radius protocol expects LOCAL_TRANSPORT_COST=pixel_l2_squared, got: ${LOCAL_TRANSPORT_COST}" >&2
      exit 1
      ;;
  esac
  case "$(printf '%s' "${LOCAL_INPUT_PGD_GEOMETRY}" | tr '[:upper:]' '[:lower:]')" in
    pixel|pixel_l2|pixel_l2_squared) ;;
    *)
      echo "[FATAL] WRM normalized-radius protocol expects LOCAL_INPUT_PGD_GEOMETRY=pixel_l2_squared, got: ${LOCAL_INPUT_PGD_GEOMETRY}" >&2
      exit 1
      ;;
  esac

  P_NORM="$(canonical_p_norm "${LOCAL_INP_P}")"
  if [ -n "${LOCAL_WRM_CP}" ]; then
    LOCAL_WRM_CP_VALUE="${LOCAL_WRM_CP}"
  else
    echo "[WRM protocol] Computing C_${P_NORM}=E_train||X_pixel||_${P_NORM} from ${LOCAL_DATA_DIR} ..." >&2
    LOCAL_WRM_CP_VALUE="$(compute_wrm_cp "${P_NORM}")"
  fi

  WRM_MODE="$(printf '%s' "${LOCAL_WRM_PGD_EPS_MODE}" | tr '[:upper:]' '[:lower:]')"
  case "${WRM_MODE}" in
    raw|epsilon|epsilon_adv)
      LOCAL_WRM_NORMALIZED_RADIUS_VALUE="$(
        WRM_EPS="${LOCAL_INP_EPS_REQUESTED}" \
        WRM_CP_VALUE="${LOCAL_WRM_CP_VALUE}" \
        python - <<'PY'
import math
import os
eps = float(os.environ["WRM_EPS"])
cp = float(os.environ["WRM_CP_VALUE"])
if not math.isfinite(eps) or eps < 0.0:
    raise SystemExit("LOCAL_INP_EPS raw PGD radius must be non-negative.")
if not math.isfinite(cp) or cp <= 0.0:
    raise SystemExit("WRM C_p must be positive.")
print(f"{eps / cp:.12g}")
PY
      )"
      LOCAL_INP_EPS_PROTOCOL_TAG="wrmraw"
      echo "[WRM protocol] LOCAL_INP_EPS raw epsilon_adv=${LOCAL_INP_EPS_REQUESTED}; C_${P_NORM}=${LOCAL_WRM_CP_VALUE}; normalized rho=epsilon_adv/C_${P_NORM}=${LOCAL_WRM_NORMALIZED_RADIUS_VALUE}" >&2
      ;;
    normalized|normalized_radius|rho)
      if ! is_truthy "${LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS}"; then
        WRM_RHO="${LOCAL_INP_EPS_REQUESTED}" \
        WRM_MAX_RHO="${LOCAL_WRM_MAX_NORMALIZED_RADIUS}" \
        python - <<'PY'
import math
import os

rho = float(os.environ["WRM_RHO"])
max_rho = float(os.environ["WRM_MAX_RHO"])
if not math.isfinite(rho) or rho < 0.0:
    raise SystemExit("LOCAL_INP_EPS normalized WRM radius must be non-negative and finite.")
if not math.isfinite(max_rho) or max_rho < 0.0:
    raise SystemExit("LOCAL_WRM_MAX_NORMALIZED_RADIUS must be non-negative and finite.")
if rho > max_rho:
    raise SystemExit(
        "LOCAL_INP_EPS is interpreted as WRM-normalized rho because "
        "LOCAL_WRM_PGD_EPS_MODE=normalized_radius, but "
        f"rho={rho:g} exceeds LOCAL_WRM_MAX_NORMALIZED_RADIUS={max_rho:g}. "
        "Use WRM figure-style values such as 0.05, 0.10, 0.15, 0.20, 0.25; "
        "or set LOCAL_WRM_ALLOW_LARGE_NORMALIZED_RADIUS=1 if this large "
        "normalized attack is intentional."
    )
PY
      fi
      LOCAL_INP_EPS="$(
        WRM_RHO="${LOCAL_INP_EPS_REQUESTED}" \
        WRM_CP_VALUE="${LOCAL_WRM_CP_VALUE}" \
        python - <<'PY'
import math
import os
rho = float(os.environ["WRM_RHO"])
cp = float(os.environ["WRM_CP_VALUE"])
if not math.isfinite(rho) or rho < 0.0:
    raise SystemExit("LOCAL_INP_EPS normalized radius must be non-negative.")
if not math.isfinite(cp) or cp <= 0.0:
    raise SystemExit("WRM C_p must be positive.")
print(f"{rho * cp:.12g}")
PY
      )"
      LOCAL_WRM_NORMALIZED_RADIUS_VALUE="${LOCAL_INP_EPS_REQUESTED}"
      LOCAL_INP_EPS_PROTOCOL_TAG="wrmrho"
      echo "[WRM protocol] LOCAL_INP_EPS normalized rho=${LOCAL_INP_EPS_REQUESTED}; C_${P_NORM}=${LOCAL_WRM_CP_VALUE}; effective PGD radius epsilon_adv=${LOCAL_INP_EPS}" >&2
      ;;
    *)
      echo "[FATAL] LOCAL_WRM_PGD_EPS_MODE must be raw or normalized_radius, got: ${LOCAL_WRM_PGD_EPS_MODE}" >&2
      exit 1
      ;;
  esac
fi

LR_TAG="$(tag_token "${LOCAL_LR_THETA}")"
LAM_TAG="$(tag_token "${LOCAL_PENALTY_LAMBDA}")"
K_TAG="$(tag_token "${LOCAL_K}")"
ALG_TAG="$(tag_token "${LOCAL_RUN_ONLY_ALGO}")"
if [ "${LOCAL_RUN_ONLY_ALGO}" = "npf" ]; then
  ACT_TAG="$(tag_token "${LOCAL_NPF_ACTIVATION}")"
  RANK_TAG="o$(tag_token "${LOCAL_NPF_OUTER_RANK}")_i$(tag_token "${LOCAL_NPF_INNER_RANK}")"
  ID_TAG="$(tag_token "${LOCAL_NPF_IDENTITY_INIT}")"
  RECT_TAG="$(tag_token "${LOCAL_NPF_POSITIVE_WEIGHT_RECTIFIER}")"
else
  ACT_TAG="$(tag_token "${LOCAL_NPF_LASTQUAD_ACTIVATION}")"
  RANK_TAG="$(tag_token "${LOCAL_NPF_LASTQUAD_OUTPUT_RANK}")"
  ID_TAG="$(tag_token "${LOCAL_NPF_LASTQUAD_IDENTITY_INIT}")"
  RECT_TAG="$(tag_token "${LOCAL_NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER}")"
fi
RESET_TAG="$(tag_token "${LOCAL_NPF_RESET_OMEGA_EACH_BATCH}")"
PROJ_TAG="$(tag_token "${LOCAL_NPF_PROJECT_TO_INPUT_BALL}")"
ANCHOR_TAG="$(tag_token "${LOCAL_NPF_PGD_ANCHOR_WEIGHT}")"
LOSS_TAG="$(tag_token "${LOCAL_INPUT_PGD_LOSS}")"
GEOM_TAG="$(tag_token "${LOCAL_INPUT_PGD_GEOMETRY}")"
EPS_REQ_TAG="$(tag_token "${LOCAL_INP_EPS_REQUESTED}")"
EPS_EFF_TAG="$(tag_token "${LOCAL_INP_EPS}")"
RHO_TAG="$(tag_token "${LOCAL_WRM_NORMALIZED_RADIUS_VALUE}")"
PGDN_TAG="$(tag_token "${LOCAL_EVAL_PGD_SAMPLES}")"
ALIGN_TAG="$(tag_token "${LOCAL_EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES}")"
TRAIN_LOSS_TAG="$(tag_token "${TRAIN_ADV_LOSS_TAG}")"
BN_TAG="bnupdate"
if [[ "${LOCAL_FREEZE_BATCHNORM}" =~ ^(1|true|True|TRUE|yes|Yes|YES)$ ]]; then
  BN_TAG="bnfrozen"
fi
ADV_MODE_TAG="adveval"
if is_falsy "${LOCAL_ADVERSARY_CLASSIFIER_EVAL}"; then
  ADV_MODE_TAG="advtrain"
fi
OPT_TAG="$(tag_token "${LOCAL_NPF_INNER_OPTIMIZER}")"
# Provenance tokens the old tag omitted (cost convention, init eps, hidden
# widths): their absence made past failed runs unreproducible from dir names.
COST_TAG="$(tag_token "${LOCAL_TRANSPORT_COST}")"
if [ "${LOCAL_RUN_ONLY_ALGO}" = "npf" ]; then
  IEPS_TAG="$(tag_token "${LOCAL_NPF_INIT_EPS}")"
  HID_TAG="$(tag_token "${LOCAL_NPF_HIDDEN}")"
else
  IEPS_TAG="$(tag_token "${LOCAL_NPF_LASTQUAD_INIT_EPS}")"
  HID_TAG="$(tag_token "${LOCAL_NPF_LASTQUAD_HIDDEN}")"
fi
RUN_TAG=${RUN_TAG:-${ALG_TAG}_ott_lam${LAM_TAG}_cost${COST_TAG}_adv${TRAIN_LOSS_TAG}_lr${LR_TAG}_K${K_TAG}_ep${LOCAL_ADV_EPOCHS}_warm${LOCAL_WARMUP_EPOCHS}_rank${RANK_TAG}_id${ID_TAG}_ieps${IEPS_TAG}_act${ACT_TAG}_rect${RECT_TAG}_hid${HID_TAG}_reset${RESET_TAG}_proj${PROJ_TAG}_anch${ANCHOR_TAG}_${BN_TAG}_${ADV_MODE_TAG}_opt${OPT_TAG}_pgd${LOSS_TAG}_${GEOM_TAG}_${LOCAL_INP_EPS_PROTOCOL_TAG}${EPS_REQ_TAG}_eff${EPS_EFF_TAG}_rho${RHO_TAG}n${PGDN_TAG}_align${ALIGN_TAG}_seed${LOCAL_SEED}}

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
RESET_PARAMETRIC_BB_EACH_BATCH="${LOCAL_RESET_PARAMETRIC_BB_EACH_BATCH}" \
PARAMETRIC_BB_MAX_GRAD_NORM="${LOCAL_PARAMETRIC_BB_MAX_GRAD_NORM}" \
NPF_OMEGA_WEIGHT_DECAY="${LOCAL_NPF_OMEGA_WEIGHT_DECAY}" \
EPOCHS_ICNN_PRETRAIN="${LOCAL_WARMUP_EPOCHS}" \
FREEZE_BATCHNORM="${LOCAL_FREEZE_BATCHNORM}" \
FREEZE_BATCHNORM_AFFINE="${LOCAL_FREEZE_BATCHNORM_AFFINE}" \
ADVERSARY_CLASSIFIER_EVAL="${LOCAL_ADVERSARY_CLASSIFIER_EVAL}" \
BATCHNORM_ONLINE_REFRESH="${LOCAL_BATCHNORM_ONLINE_REFRESH}" \
BATCHNORM_ONLINE_REFRESH_MOMENTUM="${LOCAL_BATCHNORM_ONLINE_REFRESH_MOMENTUM}" \
RECALIBRATE_BATCHNORM="${LOCAL_RECALIBRATE_BATCHNORM}" \
BATCHNORM_RECALIBRATION_BATCHES="${LOCAL_BATCHNORM_RECALIBRATION_BATCHES}" \
BATCHNORM_RECALIBRATION_RESET="${LOCAL_BATCHNORM_RECALIBRATION_RESET}" \
BATCHNORM_RECALIBRATION_MOMENTUM="${LOCAL_BATCHNORM_RECALIBRATION_MOMENTUM}" \
NPF_INNER_OPTIMIZER="${LOCAL_NPF_INNER_OPTIMIZER}" \
NPF_RESET_OMEGA_EACH_BATCH="${LOCAL_NPF_RESET_OMEGA_EACH_BATCH}" \
NPF_PROJECT_TO_INPUT_BALL="${LOCAL_NPF_PROJECT_TO_INPUT_BALL}" \
NPF_PGD_ANCHOR_WEIGHT="${LOCAL_NPF_PGD_ANCHOR_WEIGHT}" \
NPF_PGD_ANCHOR_STEPS="${LOCAL_NPF_PGD_ANCHOR_STEPS}" \
NPF_PGD_ANCHOR_RESTARTS="${LOCAL_NPF_PGD_ANCHOR_RESTARTS}" \
NPF_PGD_ANCHOR_LOSS="${LOCAL_NPF_PGD_ANCHOR_LOSS}" \
NPF_HIDDEN="${LOCAL_NPF_HIDDEN}" \
NPF_OUTER_RANK="${LOCAL_NPF_OUTER_RANK}" \
NPF_INNER_RANK="${LOCAL_NPF_INNER_RANK}" \
NPF_ACTIVATION="${LOCAL_NPF_ACTIVATION}" \
NPF_ELU_ALPHA="${LOCAL_NPF_ELU_ALPHA}" \
NPF_SOFTPLUS_BETA="${LOCAL_NPF_SOFTPLUS_BETA}" \
NPF_INIT_EPS="${LOCAL_NPF_INIT_EPS}" \
NPF_IDENTITY_INIT="${LOCAL_NPF_IDENTITY_INIT}" \
NPF_STRONG_CONVEXITY="${LOCAL_NPF_STRONG_CONVEXITY}" \
NPF_POS_WEIGHTS="${LOCAL_NPF_POS_WEIGHTS}" \
NPF_POSITIVE_WEIGHT_RECTIFIER="${LOCAL_NPF_POSITIVE_WEIGHT_RECTIFIER}" \
NPF_LASTQUAD_HIDDEN="${LOCAL_NPF_LASTQUAD_HIDDEN}" \
NPF_LASTQUAD_OUTPUT_RANK="${LOCAL_NPF_LASTQUAD_OUTPUT_RANK}" \
NPF_LASTQUAD_ACTIVATION="${LOCAL_NPF_LASTQUAD_ACTIVATION}" \
NPF_LASTQUAD_ELU_ALPHA="${LOCAL_NPF_LASTQUAD_ELU_ALPHA}" \
NPF_LASTQUAD_SOFTPLUS_BETA="${LOCAL_NPF_LASTQUAD_SOFTPLUS_BETA}" \
NPF_LASTQUAD_INIT_EPS="${LOCAL_NPF_LASTQUAD_INIT_EPS}" \
NPF_LASTQUAD_IDENTITY_INIT="${LOCAL_NPF_LASTQUAD_IDENTITY_INIT}" \
NPF_LASTQUAD_STRONG_CONVEXITY="${LOCAL_NPF_LASTQUAD_STRONG_CONVEXITY}" \
NPF_LASTQUAD_POS_WEIGHTS="${LOCAL_NPF_LASTQUAD_POS_WEIGHTS}" \
NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER="${LOCAL_NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER}" \
NPF_MUON_LR="${LOCAL_NPF_MUON_LR}" \
NPF_MUON_MOMENTUM="${LOCAL_NPF_MUON_MOMENTUM}" \
NPF_MUON_NESTEROV="${LOCAL_NPF_MUON_NESTEROV}" \
NPF_MUON_NS_STEPS="${LOCAL_NPF_MUON_NS_STEPS}" \
NPF_MUON_MATRIX_LR_SCALE="${LOCAL_NPF_MUON_MATRIX_LR_SCALE}" \
NPF_MUON_WEIGHT_DECAY="${LOCAL_NPF_MUON_WEIGHT_DECAY}" \
NPF_MUON_FALLBACK="${LOCAL_NPF_MUON_FALLBACK}" \
NPF_MUON_FALLBACK_LR="${LOCAL_NPF_MUON_FALLBACK_LR}" \
NPF_MUON_FALLBACK_WEIGHT_DECAY="${LOCAL_NPF_MUON_FALLBACK_WEIGHT_DECAY}" \
NPF_MUON_ADAM_BETA1="${LOCAL_NPF_MUON_ADAM_BETA1}" \
NPF_MUON_ADAM_BETA2="${LOCAL_NPF_MUON_ADAM_BETA2}" \
NPF_MUON_ADAM_EPS="${LOCAL_NPF_MUON_ADAM_EPS}" \
NPF_MUON_MAX_GRAD_NORM="${LOCAL_NPF_MUON_MAX_GRAD_NORM}" \
BB_ALPHA0="${LOCAL_BB_ALPHA0}" \
BB_ALPHA_MIN="${LOCAL_BB_ALPHA_MIN}" \
BB_ALPHA_MAX="${LOCAL_BB_ALPHA_MAX}" \
BB_LS_C="${LOCAL_BB_LS_C}" \
BB_LS_SHRINK="${LOCAL_BB_LS_SHRINK}" \
BB_LS_MAX_STEPS="${LOCAL_BB_LS_MAX_STEPS}" \
WRM_NORMALIZED_RADIUS_PROTOCOL="${LOCAL_WRM_NORMALIZED_RADIUS_PROTOCOL}" \
WRM_PGD_EPS_MODE="${LOCAL_WRM_PGD_EPS_MODE}" \
WRM_NORMALIZED_RADIUS_REQUESTED="${LOCAL_INP_EPS_REQUESTED}" \
WRM_NORMALIZED_RADIUS="${LOCAL_WRM_NORMALIZED_RADIUS_VALUE}" \
WRM_CP="${LOCAL_WRM_CP_VALUE}" \
WRM_EFFECTIVE_INP_EPS="${LOCAL_INP_EPS}" \
INP_P="${LOCAL_INP_P}" \
INP_EPS="${LOCAL_INP_EPS}" \
INP_STEPS="${LOCAL_INP_STEPS}" \
INP_RESTARTS="${LOCAL_INP_RESTARTS}" \
EVAL_PGD_SAMPLES="${LOCAL_EVAL_PGD_SAMPLES}" \
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
FROZEN_ADVERSARY_EPOCHS="${LOCAL_FROZEN_ADVERSARY_EPOCHS}" \
FROZEN_ADVERSARY_MAP_STEPS="${LOCAL_FROZEN_ADVERSARY_MAP_STEPS}" \
OMEGA_STEPS="${LOCAL_K}" \
RUN_ONLY_ALGO="${LOCAL_RUN_ONLY_ALGO}" \
bash run_runtime_sweep_ddp.sh \
  "${LOCAL_SPLIT}" \
  "${LOCAL_SEED}" \
  "${LOCAL_NPROC}" \
  "${LOCAL_K}" \
  "${LOCAL_ADV_EPOCHS}" \
  "${RUN_TAG}"
