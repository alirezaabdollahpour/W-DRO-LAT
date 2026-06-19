#!/usr/bin/env bash
# =============================================================================
# OOD evaluation sweep for the input-ICNN / NPF-lastquad checkpoint.
#
# By default this evaluates the best-robust checkpoint produced by
# run_runtime_sweep_ddp.sh:
#
#   input_icnn_ddp_runs/npf_lastquad_lr0p003_lam10_ce_eps0p5_K20_ep50_seed1_ddp4/
#     npf_lastquad/seed_1/npf_lastquad_seed1_best_robust.pth
#
# Evaluations:
#   * CIFAR-10 clean accuracy
#   * CIFAR-10 L2-PGD robustness in pixel space, eps=0.5
#   * CIFAR-10 AutoAttack L2, eps=0.5, attacks APGD-CE + APGD-T
#   * CIFAR-10.1 and CIFAR-10.2 clean accuracy
#   * CIFAR-10-C clean accuracy, severities 1..5
#
# Usage:
#   bash run_OOD_sweep.sh [SPLIT] [SEED] [K]
#
#   SPLIT=0: evaluate OOD_ALGOS sequentially. Default OOD_ALGOS=npf_lastquad.
#   SPLIT=1: evaluate npf_lastquad only.
#
# Other algorithms are intentionally not part of the default sweep. To compare
# extra checkpoints, pass OOD_ALGOS="npf_lastquad wrm ..." and set RESULTS_DIR
# to a tree with matching ${algo}/seed_${SEED}/checkpoint files.
# =============================================================================

set -uo pipefail

SPLIT="${1:-0}"
SEED="${2:-1}"
K="${3:-20}"

# ---- Evaluation hyperparameters ----
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"

INP_P="${INP_P:-2}"
INP_EPS="${INP_EPS:-0.5}"
INP_STEPS="${INP_STEPS:-20}"
INP_RESTARTS="${INP_RESTARTS:-5}"
INP_EPS_SWEEP_RAW="${INP_EPS_SWEEP:-${PGD_EPS_SWEEP:-}}"
INP_EPS_SWEEP_RAW="${INP_EPS_SWEEP_RAW//,/ }"
read -r -a INP_EPS_SWEEP <<< "${INP_EPS_SWEEP_RAW}"
INP_SWEEP_MAX_SAMPLES="${INP_SWEEP_MAX_SAMPLES:--1}"

RUN_AUTOATTACK="${RUN_AUTOATTACK:-1}"
AA_VERSION="${AA_VERSION:-custom}"
AA_NORM="${AA_NORM:-L2}"
AA_EPS="${AA_EPS:-0.5}"
AA_BS="${AA_BS:-128}"
AA_ATTACKS_RAW="${AA_ATTACKS:-apgd-ce apgd-t}"
AA_ATTACKS_RAW="${AA_ATTACKS_RAW//,/ }"
read -r -a AA_ATTACKS <<< "${AA_ATTACKS_RAW}"
AUTOATTACK_MAX_EXAMPLES="${AUTOATTACK_MAX_EXAMPLES:--1}"
AUTOATTACK_SEED="${AUTOATTACK_SEED:-}"
AUTOATTACK_ITERS="${AUTOATTACK_ITERS:-100}"
AUTOATTACK_RESTARTS="${AUTOATTACK_RESTARTS:-}"

SKIP_CIFAR10W="${SKIP_CIFAR10W:-1}"
SKIP_CIFAR10C="${SKIP_CIFAR10C:-0}"
CIFAR10C_DATA_DIR="${CIFAR10C_DATA_DIR:-}"
CIFAR10C_SEVERITIES_RAW="${CIFAR10C_SEVERITIES:-1 2 3 4 5}"
CIFAR10C_SEVERITIES_RAW="${CIFAR10C_SEVERITIES_RAW//,/ }"
read -r -a CIFAR10C_SEVERITIES <<< "${CIFAR10C_SEVERITIES_RAW}"
CIFAR10C_CORRUPTIONS_RAW="${CIFAR10C_CORRUPTIONS:-}"
CIFAR10C_CORRUPTIONS_RAW="${CIFAR10C_CORRUPTIONS_RAW//,/ }"
read -r -a CIFAR10C_CORRUPTIONS <<< "${CIFAR10C_CORRUPTIONS_RAW}"
CIFAR10C_MAX_EXAMPLES="${CIFAR10C_MAX_EXAMPLES:-10000}"

# ---- Paths ----
SRC_DIR="${SRC_DIR:-/mloscratch/homes/aabdolla/LAT}"
NPROC="${NPROC:-4}"
DEFAULT_RUN_NAME="npf_lastquad_lr0p003_lam10_ce_eps0p5_K${K}_ep50_seed${SEED}_ddp${NPROC}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${SRC_DIR}/input_icnn_ddp_runs/${RUN_NAME}}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_robust}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SRC_DIR}/evaluate_wrm_lat_cifar10_variants.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPS_TAG="${INP_EPS//./p}"
FORCE_EVAL="${FORCE_EVAL:-0}"
STORE_EVAL_WITH_CKPT="${STORE_EVAL_WITH_CKPT:-0}"

# Override this for a one-off checkpoint outside RESULTS_DIR.
CKPT="${CKPT:-${CKPT_OVERRIDE:-}}"
if [ "${STORE_EVAL_WITH_CKPT}" = "1" ] && [ -n "${CKPT}" ]; then
    OOD_ROOT="${OOD_ROOT:-$(dirname "${CKPT}")}"
else
    OOD_ROOT="${OOD_ROOT:-${RESULTS_DIR}/ood_eval_${CHECKPOINT_KIND}_l2eps${EPS_TAG}}"
fi
OOD_LOG_DIR="${OOD_ROOT}/logs"
OOD_MANIFEST="${OOD_ROOT}/manifest.tsv"

OOD_ALGOS_DEFAULT="npf_lastquad"
OOD_ALGOS_RAW="${OOD_ALGOS:-${OOD_ALGOS_DEFAULT}}"
OOD_ALGOS_RAW="${OOD_ALGOS_RAW//,/ }"
read -r -a OOD_ALGOS <<< "${OOD_ALGOS_RAW}"

# Optional label override for the table when evaluating one method.
METHOD_NAME="${METHOD_NAME:-ICNN-DRO}"
METHOD_LABELS="${METHOD_LABELS:-npf_lastquad=${METHOD_NAME}}"

cd "${SRC_DIR}" || exit 1
mkdir -p "${OOD_ROOT}" "${OOD_LOG_DIR}"

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[FATAL] eval script not found: ${EVAL_SCRIPT}" >&2
    exit 1
fi

if [ ! -f "${OOD_MANIFEST}" ]; then
    printf "status\talgorithm\tseed\tcheckpoint_kind\tckpt\tjson\tlog\tstarted_utc\tended_utc\telapsed_seconds\n" \
        > "${OOD_MANIFEST}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

PYTHON_EXE="$("${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
PY
)"
if [ $? -ne 0 ]; then
    echo "[FATAL] Could not run PYTHON_BIN=${PYTHON_BIN}" >&2
    exit 1
fi

if [ "${RUN_AUTOATTACK}" = "1" ]; then
    if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("autoattack")
if spec is None:
    print(f"[check] autoattack import failed for {sys.executable}")
    raise SystemExit(1)
print(f"[check] autoattack={spec.origin}")
PY
    then
        echo "[FATAL] AutoAttack is required for this sweep but is not installed in: ${PYTHON_EXE}" >&2
        echo "        Install it into the same environment with:" >&2
        echo "        ${PYTHON_BIN} -m pip install git+https://github.com/fra31/auto-attack.git" >&2
        exit 1
    fi
fi

resolve_ckpt() {
    local algo="$1"
    if [ -n "${CKPT}" ]; then
        printf "%s\n" "${CKPT}"
        return 0
    fi

    case "${CHECKPOINT_KIND}" in
        best_robust)
            printf "%s\n" "${RESULTS_DIR}/${algo}/seed_${SEED}/${algo}_seed${SEED}_best_robust.pth"
            ;;
        last)
            printf "%s\n" "${RESULTS_DIR}/${algo}/seed_${SEED}/${algo}_seed${SEED}_last.pth"
            ;;
        final)
            printf "%s\n" "${RESULTS_DIR}/${algo}/seed_${SEED}/final.pth"
            ;;
        *)
            printf "%s\n" "${CHECKPOINT_KIND}"
            ;;
    esac
}

echo ""
echo "================================================================"
echo "  OOD evaluation sweep"
echo "  SPLIT=${SPLIT}  SEED=${SEED}  K=${K}"
echo "  Algos: ${OOD_ALGOS[*]}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Checkpoint kind: ${CHECKPOINT_KIND}"
echo "  Python: ${PYTHON_EXE}"
echo "  Eval: L${INP_P}-PGD eps=${INP_EPS}, steps=${INP_STEPS}, restarts=${INP_RESTARTS}"
if [ "${#INP_EPS_SWEEP[@]}" -gt 0 ]; then
    echo "        PGD eps sweep: ${INP_EPS_SWEEP[*]} (max_samples=${INP_SWEEP_MAX_SAMPLES})"
fi
if [ "${STORE_EVAL_WITH_CKPT}" = "1" ]; then
    echo "        Output mode: checkpoint directory"
fi
if [ "${RUN_AUTOATTACK}" = "1" ]; then
    if [ "${#AA_ATTACKS[@]}" -gt 0 ]; then
        echo "        AutoAttack ${AA_NORM}, eps=${AA_EPS}, version=${AA_VERSION}, attacks=${AA_ATTACKS[*]}, iters=${AUTOATTACK_ITERS}"
    else
        echo "        AutoAttack ${AA_NORM}, eps=${AA_EPS}, version=${AA_VERSION}, iters=${AUTOATTACK_ITERS}"
    fi
else
    echo "        AutoAttack disabled"
fi
if [ "${SKIP_CIFAR10C}" = "1" ]; then
    echo "        CIFAR-10-C disabled"
else
    echo "        CIFAR-10-C severities=${CIFAR10C_SEVERITIES[*]}, max_examples=${CIFAR10C_MAX_EXAMPLES}"
fi
echo "  Output: ${OOD_ROOT}"
echo "  Started: $(date)"
echo "================================================================"

run_one() {
    local algo="$1"
    local ckpt_path
    ckpt_path="$(resolve_ckpt "${algo}")"
    local out_dir="${OOD_ROOT}/${algo}/seed_${SEED}"
    if [ "${STORE_EVAL_WITH_CKPT}" = "1" ]; then
        out_dir="$(dirname "${ckpt_path}")"
    fi
    local save_json="${out_dir}/ood_eval_${CHECKPOINT_KIND}.json"
    local log_dir="${OOD_LOG_DIR}"
    if [ "${STORE_EVAL_WITH_CKPT}" = "1" ]; then
        log_dir="${out_dir}/ood_logs"
    fi
    local log_file="${log_dir}/${algo}_seed${SEED}_${CHECKPOINT_KIND}.log"
    local done_file="${out_dir}/.completed_${CHECKPOINT_KIND}"

    if [ ! -f "${ckpt_path}" ]; then
        echo "[SKIP] ${algo}: checkpoint not found at ${ckpt_path}"
        return 0
    fi
    if [ "${FORCE_EVAL}" != "1" ] && [ -f "${done_file}" ] && [ -f "${save_json}" ]; then
        echo "[SKIP] ${algo}: already completed (${save_json}); set FORCE_EVAL=1 to rerun."
        return 0
    fi

    mkdir -p "${out_dir}" "${log_dir}"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${algo} seed=${SEED}"
    echo "  ckpt=${ckpt_path}"
    echo "  json=${save_json}"
    echo "  log=${log_file}"
    echo "  Started: $(date)"
    echo "----------------------------------------------------------------"

    local eval_args=(
        "${EVAL_SCRIPT}"
        --ckpt "${ckpt_path}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --inp-p "${INP_P}"
        --inp-eps "${INP_EPS}"
        --inp-steps "${INP_STEPS}"
        --inp-restarts "${INP_RESTARTS}"
        --save-json "${save_json}"
    )
    if [ "${#INP_EPS_SWEEP[@]}" -gt 0 ]; then
        eval_args+=(--inp-eps-sweep "${INP_EPS_SWEEP[@]}")
        eval_args+=(--inp-sweep-max-samples "${INP_SWEEP_MAX_SAMPLES}")
    fi

    if [ "${SKIP_CIFAR10W}" = "1" ]; then
        eval_args+=(--skip-cifar10w)
    fi

    if [ "${SKIP_CIFAR10C}" = "1" ]; then
        eval_args+=(--skip-cifar10c)
    else
        eval_args+=(
            --cifar10c-severities "${CIFAR10C_SEVERITIES[@]}"
            --cifar10c-max-examples "${CIFAR10C_MAX_EXAMPLES}"
        )
        if [ -n "${CIFAR10C_DATA_DIR}" ]; then
            eval_args+=(--cifar10c-data-dir "${CIFAR10C_DATA_DIR}")
        fi
        if [ "${#CIFAR10C_CORRUPTIONS[@]}" -gt 0 ]; then
            eval_args+=(--cifar10c-corruptions "${CIFAR10C_CORRUPTIONS[@]}")
        fi
    fi

    if [ "${RUN_AUTOATTACK}" = "1" ]; then
        eval_args+=(
            --autoattack
            --autoattack-bs "${AA_BS}"
            --autoattack-version "${AA_VERSION}"
            --autoattack-norm "${AA_NORM}"
            --autoattack-eps "${AA_EPS}"
        )
        if [ "${#AA_ATTACKS[@]}" -gt 0 ]; then
            eval_args+=(--autoattack-attacks "${AA_ATTACKS[@]}")
        fi
        if [ "${AUTOATTACK_MAX_EXAMPLES}" != "-1" ]; then
            eval_args+=(--autoattack-max-examples "${AUTOATTACK_MAX_EXAMPLES}")
        fi
        if [ -n "${AUTOATTACK_SEED}" ]; then
            eval_args+=(--autoattack-seed "${AUTOATTACK_SEED}")
        fi
        if [ -n "${AUTOATTACK_ITERS}" ]; then
            eval_args+=(--autoattack-iters "${AUTOATTACK_ITERS}")
        fi
        if [ -n "${AUTOATTACK_RESTARTS}" ]; then
            eval_args+=(--autoattack-restarts "${AUTOATTACK_RESTARTS}")
        fi
    fi

    local started_utc ended_utc t0 t1 elapsed status
    started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    t0="$(date +%s)"

    "${PYTHON_BIN}" "${eval_args[@]}" 2>&1 | tee "${log_file}"
    status="${PIPESTATUS[0]}"

    t1="$(date +%s)"
    ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    elapsed=$((t1 - t0))

    if [ "${status}" -eq 0 ]; then
        touch "${done_file}"
        printf "completed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${algo}" "${SEED}" "${CHECKPOINT_KIND}" "${ckpt_path}" "${save_json}" "${log_file}" \
            "${started_utc}" "${ended_utc}" "${elapsed}" >> "${OOD_MANIFEST}"
        echo "[OK] ${algo} seed=${SEED} elapsed=${elapsed}s -> ${save_json}"
    else
        printf "failed_%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${status}" "${algo}" "${SEED}" "${CHECKPOINT_KIND}" "${ckpt_path}" "${save_json}" "${log_file}" \
            "${started_utc}" "${ended_utc}" "${elapsed}" >> "${OOD_MANIFEST}"
        echo "[FAIL] ${algo} seed=${SEED} status=${status}; see ${log_file}"
    fi

    sleep 2
}

case "${SPLIT}" in
    0)
        for algo in "${OOD_ALGOS[@]}"; do
            run_one "${algo}"
        done
        ;;
    1)
        run_one "npf_lastquad"
        ;;
    *)
        echo "[FATAL] Unknown SPLIT=${SPLIT}. Use 0 for OOD_ALGOS or 1 for npf_lastquad." >&2
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  OOD sweep split ${SPLIT} complete at $(date)"
echo "  Manifest: ${OOD_MANIFEST}"
echo "================================================================"

SPLIT="${SPLIT}" \
SEED="${SEED}" \
OOD_ROOT="${OOD_ROOT}" \
OOD_ALGOS="${OOD_ALGOS[*]}" \
CHECKPOINT_KIND="${CHECKPOINT_KIND}" \
METHOD_LABELS="${METHOD_LABELS}" \
INP_EPS="${INP_EPS}" \
AA_EPS="${AA_EPS}" \
STORE_EVAL_WITH_CKPT="${STORE_EVAL_WITH_CKPT}" \
CKPT="${CKPT}" \
"${PYTHON_BIN}" - << 'PYTHON_AGG'
import csv
import json
import math
import os
from pathlib import Path

OOD_ROOT = Path(os.environ["OOD_ROOT"])
SEED = int(os.environ["SEED"])
ALGOS = os.environ["OOD_ALGOS"].split()
CHECKPOINT_KIND = os.environ["CHECKPOINT_KIND"]
INP_EPS = os.environ["INP_EPS"]
AA_EPS = os.environ["AA_EPS"]
STORE_EVAL_WITH_CKPT = os.environ.get("STORE_EVAL_WITH_CKPT", "0") == "1"
CKPT = os.environ.get("CKPT", "")

DEFAULT_LABELS = {
    "npf_lastquad": "ICNN-DRO",
    "npf": "NPF",
    "wrm": "WRM",
    "madry": "Madry",
    "nn_dro": "NN-DRO",
    "dual": "Dual",
    "new_ppa": "New-PPA",
    "wfr": "WFR",
}

labels = dict(DEFAULT_LABELS)
for item in os.environ.get("METHOD_LABELS", "").replace(",", " ").split():
    if "=" in item:
        key, value = item.split("=", 1)
    elif ":" in item:
        key, value = item.split(":", 1)
    else:
        continue
    labels[key.strip()] = value.strip()

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def get(d, *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def cifar10c_by_severity(scores):
    c10c = scores.get("cifar10c")
    out = {str(i): None for i in range(1, 6)}
    if not isinstance(c10c, dict):
        return out, None
    corruptions = c10c.get("corruptions")
    if isinstance(corruptions, dict):
        severities = [str(s) for s in c10c.get("severities", [1, 2, 3, 4, 5])]
        for sev in severities:
            vals = []
            for corr_scores in corruptions.values():
                if isinstance(corr_scores, dict) and is_number(corr_scores.get(sev)):
                    vals.append(float(corr_scores[sev]))
            if vals:
                out[sev] = sum(vals) / len(vals)
        avg_vals = [v for v in out.values() if is_number(v)]
        avg = sum(avg_vals) / len(avg_vals) if avg_vals else None
        return out, avg
    if is_number(c10c.get("mean")):
        return out, float(c10c["mean"])
    return out, None

def load_row(algo):
    if STORE_EVAL_WITH_CKPT and CKPT:
        path = Path(CKPT).parent / f"ood_eval_{CHECKPOINT_KIND}.json"
        fallback = Path(CKPT).parent / "ood_eval.json"
    else:
        path = OOD_ROOT / algo / f"seed_{SEED}" / f"ood_eval_{CHECKPOINT_KIND}.json"
        fallback = OOD_ROOT / algo / f"seed_{SEED}" / "ood_eval.json"
    if not path.exists() and fallback.exists():
        path = fallback
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    scores = data.get("scores", {})
    c10c_sev, c10c_avg = cifar10c_by_severity(scores)
    return {
        "algo": algo,
        "method": labels.get(algo, algo),
        "json": str(path),
        "ckpt": data.get("ckpt"),
        "cifar10_clean": scores.get("cifar10", scores.get("cifar10_test")),
        "cifar10_pgd": get(scores, "cifar10_pgd", "acc", default=get(scores, "cifar10_test_pgd", "acc")),
        "cifar10_aa": get(scores, "autoattack", "acc", default=get(scores, "cifar10_autoattack", "acc")),
        "cifar10_1_clean": scores.get("cifar10.1_v6"),
        "cifar10_2_clean": scores.get("cifar10.2_test"),
        "cifar10c_s1": c10c_sev["1"],
        "cifar10c_s2": c10c_sev["2"],
        "cifar10c_s3": c10c_sev["3"],
        "cifar10c_s4": c10c_sev["4"],
        "cifar10c_s5": c10c_sev["5"],
        "cifar10c_avg": c10c_avg,
    }

rows = [row for algo in ALGOS if (row := load_row(algo)) is not None]
if not rows:
    print("[agg] no completed OOD evaluations to summarize yet.")
    raise SystemExit(0)

metric_cols = [
    "cifar10_clean",
    "cifar10_pgd",
    "cifar10_aa",
    "cifar10_1_clean",
    "cifar10_2_clean",
    "cifar10c_s1",
    "cifar10c_s2",
    "cifar10c_s3",
    "cifar10c_s4",
    "cifar10c_s5",
    "cifar10c_avg",
]

best = {}
for col in metric_cols:
    vals = [float(row[col]) for row in rows if is_number(row.get(col))]
    best[col] = max(vals) if vals else None

def fmt_plain(x):
    return "--" if not is_number(x) else f"{float(x):.2f}"

def fmt_latex(row, col):
    x = row.get(col)
    if not is_number(x):
        return "--"
    value = float(x)
    cell = f"{value:.2f}"
    if best.get(col) is not None and abs(value - best[col]) < 5e-3:
        return r"\textbf{" + cell + "}"
    return cell

summary_path = OOD_ROOT / f"summary_seed{SEED}.json"
summary_path.write_text(json.dumps({row["algo"]: row for row in rows}, indent=2))

csv_path = OOD_ROOT / f"table_seed{SEED}.csv"
csv_cols = ["method", "algo", *metric_cols, "ckpt", "json"]
with csv_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=csv_cols)
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col) for col in csv_cols})

latex_path = OOD_ROOT / f"table_seed{SEED}.tex"
latex_lines = [
    r"\begin{table*}[!t]",
    r"    \centering",
    r"    \renewcommand{\arraystretch}{1.15}",
    r"    \caption{Accuracy (\%) under adversarial attacks, natural domain shifts and image corruptions. The left section reports results for the original CIFAR-10 test set (Clean, PGD, and AA). The center columns evaluate accuracy on the unseen domains of CIFAR-10.1 and 10.2. The right section (CIFAR-10-C) details performance across five increasing levels of corruption severity, with the final column representing the mean accuracy across all five levels. Bold values indicate the best results.}",
    r"    \vspace{-0.25cm}",
    r"    \resizebox{\linewidth}{!}{",
    r"        \begin{tabular}{l ccc c c cccccc}",
    r"            \toprule",
    r"            \multirow{2.5}{*}{Method} & \multicolumn{3}{c}{CIFAR-10} & \multicolumn{1}{c}{CIFAR-10.1} & \multicolumn{1}{c}{CIFAR-10.2} & \multicolumn{6}{c}{CIFAR-10-C (Common Corruption)} \\",
    r"            \cmidrule(lr){2-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-12}",
    f"            & Clean & PGD$_{{\\varepsilon={INP_EPS}}}$ & AA$_{{\\varepsilon={AA_EPS}}}$ & Clean & Clean & 1 & 2 & 3 & 4 & 5 & Avg \\\\",
    r"            \midrule",
]
for row in rows:
    cells = [
        row["method"],
        fmt_latex(row, "cifar10_clean"),
        fmt_latex(row, "cifar10_pgd"),
        fmt_latex(row, "cifar10_aa"),
        fmt_latex(row, "cifar10_1_clean"),
        fmt_latex(row, "cifar10_2_clean"),
        fmt_latex(row, "cifar10c_s1"),
        fmt_latex(row, "cifar10c_s2"),
        fmt_latex(row, "cifar10c_s3"),
        fmt_latex(row, "cifar10c_s4"),
        fmt_latex(row, "cifar10c_s5"),
        fmt_latex(row, "cifar10c_avg"),
    ]
    latex_lines.append("            " + " & ".join(cells) + r" \\")
latex_lines.extend([
    r"            \bottomrule",
    r"        \end{tabular}",
    r"    }",
    r"    \label{tab:cifar10-cifar101-cifar102-ood}",
    r"    \vspace{-0.7cm}",
    r"\end{table*}",
    "",
])
latex_path.write_text("\n".join(latex_lines))

print()
print("=" * 132)
print(f"  OOD evaluation summary (seed={SEED}, root={OOD_ROOT})")
print("=" * 132)
header = (
    f"{'Method':<14} {'Clean':>8} {'PGD':>8} {'AA':>8} "
    f"{'10.1':>8} {'10.2':>8} {'C-s1':>8} {'C-s2':>8} {'C-s3':>8} "
    f"{'C-s4':>8} {'C-s5':>8} {'C-Avg':>8}"
)
print(header)
print("-" * len(header))
for row in rows:
    print(
        f"{row['method']:<14} "
        f"{fmt_plain(row['cifar10_clean']):>8} "
        f"{fmt_plain(row['cifar10_pgd']):>8} "
        f"{fmt_plain(row['cifar10_aa']):>8} "
        f"{fmt_plain(row['cifar10_1_clean']):>8} "
        f"{fmt_plain(row['cifar10_2_clean']):>8} "
        f"{fmt_plain(row['cifar10c_s1']):>8} "
        f"{fmt_plain(row['cifar10c_s2']):>8} "
        f"{fmt_plain(row['cifar10c_s3']):>8} "
        f"{fmt_plain(row['cifar10c_s4']):>8} "
        f"{fmt_plain(row['cifar10c_s5']):>8} "
        f"{fmt_plain(row['cifar10c_avg']):>8}"
    )
print(f"\nSaved raw summary: {summary_path}")
print(f"Saved CSV table:   {csv_path}")
print(f"Saved LaTeX table: {latex_path}")
PYTHON_AGG
