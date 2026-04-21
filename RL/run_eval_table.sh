#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# ---- Configuration ----
# Evaluate ALL method checkpoints produced by a single training run of
# RL_minimal.py, so the comparison is fair (same seed, same environment,
# same eval settings, same timestamp prefix => same training schedule).
#
# Each training run produces files of the form
#   RL_minimal_cartpole_<tag>_seed<S>_lam_<L>_softplusbeta_<B>_<TS>_<method>_policy.pt
# where <TS> is a UTC timestamp shared across all methods in that run.
#
# We auto-discover the most recent timestamp that has at least one policy
# and include every *_<method>_policy.pt file sharing that prefix.

# Preferred presentation order. Nominal first (baseline), then particle-family
# and WDRO baselines, ending with the paper's main method (ICNN).
ORDER=(nominal particle svg wgf wfr rgo ppa new_ppa dual npf nn_dro algo1 icnn)

# Optional: pin a specific run prefix (everything before "_<method>_policy.pt").
# Example: PREFIX="RL_minimal_cartpole_custom_seed0_lam_3.0_softplusbeta_20.0_20260419T163436Z"
PREFIX="${PREFIX:-}"

if [ -z "$PREFIX" ]; then
    # Latest checkpoint file (any method); strip a known method suffix to
    # recover the run prefix. We must match against the method list because
    # the prefix itself contains underscores (e.g. the timestamp chunk).
    LATEST=$(ls -t "${SCRIPT_DIR}"/*_policy.pt 2>/dev/null | head -1 || true)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No *_policy.pt checkpoints found in ${SCRIPT_DIR}."
        echo "Train first with: bash run_RL_minimal_icnn.sh"
        exit 1
    fi
    BASE=$(basename "$LATEST" .pt)        # strip .pt
    BASE="${BASE%_policy}"                 # strip trailing _policy
    # Try longest method names first so e.g. "new_ppa" beats "ppa".
    CANDIDATES=(new_ppa nn_dro nominal particle icnn algo1 dual svg wgf wfr rgo npf ppa)
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

# Human-readable column labels used in the printed / LaTeX table.
declare -A LABEL=(
    [nominal]="Nominal"
    [particle]="Particle"
    [svg]="SVGD"
    [wgf]="WGF"
    [wfr]="WFR"
    [rgo]="RGO"
    [ppa]="PPA"
    [new_ppa]="PPA*"
    [dual]="Dual"
    [npf]="NPF"
    [nn_dro]="NN-DRO"
    [algo1]="Algo1"
    [icnn]="ICNN (WDRO)"
)

CKPTS=()
NAMES=()

for m in "${ORDER[@]}"; do
    f="${SCRIPT_DIR}/${PREFIX}_${m}_policy.pt"
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

OUT_TAG="${PREFIX}"

exec "${PYTHON_BIN}" RL_eval_table.py \
    --checkpoints   "${CKPTS[@]}" \
    --column-names  "${NAMES[@]}" \
    --trials 1000 \
    --max-steps 500 \
    --seed 42 \
    --out-latex "eval_table_${OUT_TAG}.tex" \
    --out-json  "eval_table_${OUT_TAG}.json" \
    "$@"
