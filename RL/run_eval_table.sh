#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# Where to look for *_policy.pt checkpoints. Defaults to SCRIPT_DIR (legacy:
# RL_minimal.py drops checkpoints next to itself). Override to evaluate
# checkpoints stored in a sibling dir, e.g. monge_gap_runs/horizon_1000/.
CKPT_DIR="${CKPT_DIR:-${SCRIPT_DIR}}"

# Eval-time episode cap and trial count. For the H=1000 table, leave
# MAX_STEPS at 1000. Override if you intentionally want a different cap.
MAX_STEPS="${MAX_STEPS:-1000}"
TRIALS="${TRIALS:-1000}"

# Where to write the per-prefix .tex/.json. Defaults to CKPT_DIR so eval
# artifacts land alongside the inputs they describe.
OUT_DIR="${OUT_DIR:-${CKPT_DIR}}"
mkdir -p "${OUT_DIR}"

# ---- Configuration ----
# Evaluate method checkpoints produced by a single training run of
# RL_minimal.py, so the comparison is fair (same seed, same environment,
# same eval settings, same prefix => same training schedule).
#
# Each training run produces files of the form
#   RL_minimal_cartpole_<tag>_seed<S>_lam_<L>_softplusbeta_<B>_<TS>_<method>_policy.pt
# where <TS> is a UTC timestamp shared across all methods in that run.
#
# We auto-discover the most recent prefix that has at least one policy and
# include the methods in $METHODS sharing that prefix.

# Paper-table presentation order:
# ERM, PA, RO, WFR, SDRO, NN-DRO, MPA, ICNN-DRO.
# Override, e.g. METHODS="nominal ro" bash run_eval_table.sh
METHODS="${METHODS:-nominal particle ro wfr dual nn_dro new_ppa npf}"

# Optional: pin a specific run prefix (everything before "_<method>_policy.pt").
# Example: PREFIX="RL_minimal_cartpole_custom_seed0_lam_3.0_softplusbeta_20.0_20260419T163436Z"
PREFIX="${PREFIX:-}"

if [ -z "$PREFIX" ]; then
    # Latest checkpoint file (any method); strip a known method suffix to
    # recover the run prefix. We must match against the method list because
    # the prefix itself contains underscores (e.g. the timestamp chunk).
    LATEST=$(ls -t "${CKPT_DIR}"/*_policy.pt 2>/dev/null | head -1 || true)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No *_policy.pt checkpoints found in ${CKPT_DIR}."
        echo "Train first with: bash run_RL_minimal_icnn.sh"
        exit 1
    fi
    BASE=$(basename "$LATEST" .pt)        # strip .pt
    BASE="${BASE%_policy}"                 # strip trailing _policy
    # Try longest method names first so e.g. "new_ppa" beats "ppa".
    CANDIDATES=(new_ppa nn_dro nominal particle icnn algo1 dual svg wgf wfr rgo npf ppa ro)
    for m in "${CANDIDATES[@]}"; do
        if [[ "$BASE" == *_"$m" ]]; then
            PREFIX="${BASE%_"$m"}"
            break
        fi
    done
    if [ -z "$PREFIX" ]; then
        echo "ERROR: Could not infer run prefix from $(basename "$LATEST")."
        echo "Set PREFIX=... explicitly."
        exit 1
    fi
fi

echo "Run prefix: ${PREFIX}"

# Human-readable method labels used in the printed / LaTeX table.
declare -A LABEL=(
    [nominal]="ERM"
    [ro]="RO"
    [particle]="PA"
    [svg]="SVGD"
    [wgf]="WGF"
    [wfr]="WFR"
    [rgo]="RGO"
    [ppa]="PPA"
    [new_ppa]="MPA"
    [dual]="SDRO"
    [npf]="ICNN-DRO"
    [nn_dro]="NN-DRO"
    [algo1]="WRM"
    [icnn]="ICNN"
)

CKPTS=()
NAMES=()

for m in ${METHODS}; do
    if [ -z "${LABEL[$m]+x}" ]; then
        echo "ERROR: unknown method key in METHODS: ${m}" >&2
        exit 1
    fi
    f="${CKPT_DIR}/${PREFIX}_${m}_policy.pt"
    if [ -f "$f" ]; then
        CKPTS+=("$f")
        NAMES+=("${LABEL[$m]}")
        printf "  [found] %-10s -> %s\n" "${LABEL[$m]}" "$(basename "$f")"
    else
        printf "  [skip ] %-10s (no file)\n" "${LABEL[$m]}"
    fi
done

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints matched prefix ${PREFIX}."
    exit 1
fi

echo
echo "Evaluating ${#CKPTS[@]} policies."
echo

OUT_TAG="${PREFIX}_h${MAX_STEPS}"

exec "${PYTHON_BIN}" RL_eval_table.py \
    --checkpoints   "${CKPTS[@]}" \
    --column-names  "${NAMES[@]}" \
    --trials "${TRIALS}" \
    --max-steps "${MAX_STEPS}" \
    --seed 42 \
    --layout method-rows \
    --out-latex "${OUT_DIR}/eval_table_${OUT_TAG}.tex" \
    --out-json  "${OUT_DIR}/eval_table_${OUT_TAG}.json" \
    "$@"
