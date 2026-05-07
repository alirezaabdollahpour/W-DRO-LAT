#!/bin/bash
# =============================================================================
# Input-ICNN Runtime Sweep — DDP edition
# =============================================================================
#
# Fair runtime comparison across 7 input-space adversarial training
# algorithms on CIFAR-10, with each run distributed across NPROC GPUs
# via torchrun + DistributedDataParallel.
#
# Adversaries:
#   * NPF      — NPF ICNN potential, BB+Armijo on ω parameters
#   * NN-DRO   — vanilla MLP adversary, BB+Armijo on ω parameters
#                (replaces the legacy Adam ascent)
#   * Madry/RO — fixed-step pixel-space l2-PGD threat model with
#                epsilon-ball projection + clamp; no lambda penalty
#   * WRM      — WRM inner objective on z, BB+Armijo step rule
#                (replaces constant-LR ascent)
#   * WFR      — particle ensemble + reweighting, BB+Armijo on the
#                deterministic gradient component, Langevin noise added
#                after (replaces constant-LR Langevin)
#   * New_PPA  — BB+Armijo ascent inside each round (replaces WRM ascent),
#                free-weight projection rounds unchanged
#   * Dual     — Sinkhorn entropic dual. Two operating modes:
#                  – DUAL_LANGEVIN_STEPS=0: one-shot Gaussian estimator
#                    (Gaussian particles around x). Fast but the
#                    samples come from the prior, not the Gibbs target,
#                    so the inner integral isn't actually solved.
#                  – DUAL_LANGEVIN_STEPS>0: Option-D rigorous inner
#                    sampling. Each particle is refined via a Langevin
#                    chain on the Gibbs target π(z|x) ∝ exp(U(z)),
#                    optionally MH-corrected (DUAL_MALA=1, default) for
#                    unbiased samples. The Sinkhorn dual loss is then
#                    computed on the refined particles. The one-shot
#                    estimator is recovered at K=0.
#
# All BB+Armijo hyperparameters (alpha0, alpha_min, alpha_max, ls_c,
# ls_shrink, ls_max_steps) are SHARED across the DRO inner-ascent
# methods, defaulting to NPF's settings from the LR-CIFAR10 reference.
# Madry/RO is intentionally excluded: it is standard pixel-space l2-PGD with
# epsilon, not a lambda-penalised DRO objective.
#
# ---- Parallelism strategy ----
#
# Each algorithm uses DDP across NPROC GPUs (default 4). The classifier
# is wrapped in DDP; on classifier_update the gradient all-reduce fires
# automatically. Inner adversary loops use the unwrapped module so DDP's
# reducer state isn't corrupted by forward-without-backward.
#
# For the parametric adversaries (NPF, NN-DRO) the BB+Armijo step
# all-reduces ω-gradients across ranks so every rank picks the same
# Armijo trial step. For the input-space DRO adversaries (WRM/WFR/PPA)
# each rank runs an independent BB+Armijo ascent on its local batch
# shard — no cross-rank sync is meaningful since z lives on the shard.
#
# Per-rank batch_size = BATCH_SIZE / NPROC, so the GLOBAL effective batch
# matches the single-GPU configuration for fair throughput comparison.
#
# Algorithms are run sequentially within a script invocation. To run
# subsets on multiple jobs concurrently, use the SPLIT mechanism below.
#
# ---- Fairness rules baked in ----
#
#   * Same outer hyperparameters (epochs, lr_theta, λ for DRO methods,
#     global batch). Madry/RO uses epsilon instead of λ.
#   * Same K=20 inner-iteration budget (per-algorithm interpretation
#     applied: NPF/NN-DRO omega_steps=K, WRM/WFR inner_steps=K,
#     Madry/RO pgd_steps=K, New_PPA round0_steps=refine_steps=K, Dual
#     langevin_steps=K via the Option-D MALA sampler).
#   * Same BB+Armijo step rule and hyperparameters across the DRO
#     inner-ascent methods. Madry/RO uses pixel-space l2-PGD; Dual uses the
#     Langevin/MALA sampler for its entropic Gibbs target.
#   * --skip-pgd-during-train: PGD eval would otherwise dominate.
#   * --benchmark-mode: cuDNN autotune + TF32, identical for all algos.
#   * Sequential (no in-script concurrency on the same GPUs).
#   * Per-epoch CUDA-synced wallclock; analyzer drops epoch 1.
#
# ---- Memory safety ----
#
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (less fragmentation).
#   * Per-algorithm BATCH override paths exist but unused at default config.
#   * 2-level OOM retry halves batch_size and proportionally adjusts inner
#     budget — preserves global throughput at the cost of GPU memory pressure.
#
# Usage:
#   bash run_runtime_sweep_ddp.sh [SPLIT] [SEED] [NPROC] [K] [EPOCHS]
#
#   SPLIT 0  — run all 7 algorithms sequentially (single job)
#   SPLIT 1  — npf
#   SPLIT 2  — nn_dro, madry
#   SPLIT 3  — wrm, wfr
#   SPLIT 4  — dual, new_ppa
#
# Cluster submission example (one 4-GPU job per split, splits in parallel):
#   for i in 1 2 3 4; do
#     ./csub.py -n input-icnn-ddp-$i -g 4 -t 1d --train \
#       --command "cd /path/to/LAT && bash run_runtime_sweep_ddp.sh \$i 1 4 20 30"
#   done
# =============================================================================

# Don't use set -e — we want OOM retry on individual algorithms.

SPLIT=${1:-0}
SEED=${2:-1}
NPROC=${3:-4}
K=${4:-20}
EPOCHS_ADV=${5:-30}

# ---- Paths (override with env vars) ----
SRC_DIR="${SRC_DIR:-/mnt/lts4/scratch/students/aabdolla/LAT}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${SRC_DIR}/ResNet_checkpoints/R2.pth}"
DATA_DIR="${DATA_DIR:-${SRC_DIR}/data}"
RESULTS_DIR="${RESULTS_DIR:-${SRC_DIR}/runtime_sweep_K${K}_ddp${NPROC}}"
LOG_DIR="${RESULTS_DIR}/logs"
TIME_DIR="${RESULTS_DIR}/time"
MANIFEST="${RESULTS_DIR}/run_manifest.tsv"

cd "$SRC_DIR"

# Memory + thread safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$TIME_DIR"
if [ ! -f "$MANIFEST" ]; then
    printf "status\talgorithm\tseed\tworld_size\tk\tepochs\tstarted_utc\tended_utc\telapsed_seconds\tlog_file\ttime_file\tcsv\n" > "$MANIFEST"
fi

# ---- Shared training hyperparameters ----
COMMON_BATCH=${COMMON_BATCH:-512}     # global effective batch (split across ranks)
PENALTY_LAMBDA=${PENALTY_LAMBDA:-30}
LR_THETA=${LR_THETA:-0.1}
INP_P=${INP_P:-2}
INP_EPS=${INP_EPS:-0.5}
MADRY_PGD_STEP_SIZE=${MADRY_PGD_STEP_SIZE:-0}  # 0 => trainer default 2*epsilon/K
USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
# Conservative default — cluster containers typically ship with a tiny
# /dev/shm (Docker's 64 MB default) and 4 workers/rank × 4 ranks = 16
# worker processes is enough to exhaust it. CIFAR-10 fits in RAM and the
# GPU adversary work dominates wallclock, so 2 workers/rank is plenty.
# Set NUM_WORKERS=0 if you still hit "No space left on device".
NUM_WORKERS=${NUM_WORKERS:-2}
EPOCHS_ICNN_PRETRAIN=${EPOCHS_ICNN_PRETRAIN:-0}

# ---- Shared BB+Armijo step rule (overridable; defaults match NPF's) ----
BB_ALPHA0=${BB_ALPHA0:-2e-4}
BB_ALPHA_MIN=${BB_ALPHA_MIN:-1e-7}
BB_ALPHA_MAX=${BB_ALPHA_MAX:-0.25}
BB_LS_C=${BB_LS_C:-1e-4}
BB_LS_SHRINK=${BB_LS_SHRINK:-0.5}
BB_LS_MAX_STEPS=${BB_LS_MAX_STEPS:-15}
BB_ARGS=(
  --bb-alpha0 "${BB_ALPHA0}"
  --bb-alpha-min "${BB_ALPHA_MIN}"
  --bb-alpha-max "${BB_ALPHA_MAX}"
  --bb-ls-c "${BB_LS_C}"
  --bb-ls-shrink "${BB_LS_SHRINK}"
  --bb-ls-max-steps "${BB_LS_MAX_STEPS}"
)

# ---- Dual Option-D Langevin / MALA inner sampler ----
# DUAL_LANGEVIN_STEPS=0 keeps the one-shot Gaussian estimator
# (closed-form Sinkhorn dual). >0 enables iterative MCMC refinement of
# each particle on the Gibbs target. Defaults below assume DUAL_EPSILON=0.05;
# η ≈ ε is a good starting point — MALA's accept rate is the right
# tuning signal (target ~0.5–0.8). Set DUAL_MALA=0 for plain ULA
# (cheaper, biased).
DUAL_EPSILON=${DUAL_EPSILON:-0.05}
DUAL_LANGEVIN_STEPS=${DUAL_LANGEVIN_STEPS:-${K}}
DUAL_LANGEVIN_STEP_SIZE=${DUAL_LANGEVIN_STEP_SIZE:-5e-3}
DUAL_MALA=${DUAL_MALA:-1}
DUAL_BURN_IN=${DUAL_BURN_IN:-0}
DUAL_SAMPLE_LEVEL=${DUAL_SAMPLE_LEVEL:-3}  # m=8 — keeps wallclock comparable

# ---- Pre-flight ----
echo ""
echo "================================================================"
echo "  Input-ICNN Runtime Sweep — DDP"
echo "  SPLIT=${SPLIT}  SEED=${SEED}  NPROC=${NPROC}  K=${K}"
echo "  Epochs=${EPOCHS_ADV}  Batch=${COMMON_BATCH} (global)  λ=${PENALTY_LAMBDA}"
echo "  Margin loss: ${USE_MARGIN_LOSS}"
echo "  BB+Armijo:   alpha0=${BB_ALPHA0}  alpha=[${BB_ALPHA_MIN}, ${BB_ALPHA_MAX}]"
echo "                ls_c=${BB_LS_C}  shrink=${BB_LS_SHRINK}  ls_max=${BB_LS_MAX_STEPS}"
echo "  Madry/RO:    pixel l2-PGD eps=${INP_EPS}  step_size=${MADRY_PGD_STEP_SIZE:-0} (0=auto)"
if [ "${DUAL_LANGEVIN_STEPS}" -gt 0 ]; then
    echo "  Dual mode:   Option-D Langevin/MALA sampler"
    echo "                eps=${DUAL_EPSILON}  m=2^${DUAL_SAMPLE_LEVEL}  K_langevin=${DUAL_LANGEVIN_STEPS}"
    echo "                eta=${DUAL_LANGEVIN_STEP_SIZE}  MALA=${DUAL_MALA}  burn_in=${DUAL_BURN_IN}"
else
    echo "  Dual mode:   one-shot Gaussian (closed-form Sinkhorn)"
fi
echo "  Results: ${RESULTS_DIR}"
echo "  Started: $(date)"
echo "================================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU pre-check:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo
fi

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "[FATAL] PRETRAINED_PATH not found: $PRETRAINED_PATH"
    exit 1
fi


# ======================================================================
# Per-algorithm config keyed off K (the shared inner-iteration budget).
# ======================================================================
algo_args() {
    # Per-algorithm flags. The shared --bb-* hyperparameters are added
    # by the runner outside of this function so every adversary uses
    # the SAME BB+Armijo step rule for DRO ascent methods. Madry/RO is
    # standard pixel-space l2-PGD, so --madry-pgd-step-size is honoured when set
    # (>0) and otherwise defaults to 2*epsilon/K.
    local algo="$1" k="$2"
    case "$algo" in
        npf)
            echo "--omega-steps-per-batch ${k} \
                --npf-hidden 1024 512 512 256 128 64 \
                --npf-outer-rank 8 --npf-inner-rank 2 \
                --npf-activation softplus --npf-softplus-beta 10.0 \
                --npf-init-eps 1e-4 --npf-strong-convexity 1.0"
            ;;
        nn_dro)
            echo "--omega-steps-per-batch ${k} \
                --nn-dro-hidden 512 512 256 256 128 \
                --nn-dro-activation relu --nn-dro-softplus-beta 20.0 \
                --nn-dro-init-scale 1e-3"
            ;;
        madry)
            echo "--madry-epsilon ${INP_EPS} --madry-pgd-steps ${k} \
                --madry-pgd-step-size ${MADRY_PGD_STEP_SIZE} --madry-pgd-restarts 1"
            ;;
        wrm)
            echo "--wrm-inner-steps ${k}"
            ;;
        wfr)
            echo "--wfr-inner-steps ${k} --wfr-num-samples 8 --wfr-epsilon 0.1"
            ;;
        dual)
            # Option-D rigorous inner sampling: each particle is refined
            # via a Langevin chain on the Gibbs target before the
            # Sinkhorn dual loss. With DUAL_LANGEVIN_STEPS=0 the run
            # falls back to the one-shot closed-form estimator
            # (no inner ascent — Dual is then the only method without
            # iterative inner work, kept for back-compat and as a
            # fast-but-biased reference).
            local mala_flag="--no-dual-mala"
            if [ "$DUAL_MALA" = "1" ]; then mala_flag="--dual-mala"; fi
            echo "--dual-epsilon ${DUAL_EPSILON} \
                --dual-sample-level ${DUAL_SAMPLE_LEVEL} \
                --dual-langevin-steps ${DUAL_LANGEVIN_STEPS} \
                --dual-langevin-step-size ${DUAL_LANGEVIN_STEP_SIZE} \
                --dual-burn-in ${DUAL_BURN_IN} \
                ${mala_flag}"
            ;;
        new_ppa)
            echo "--ppa-num-rounds 5 --ppa-min-rounds 2 \
                --ppa-round0-steps ${k} \
                --ppa-refine-steps ${k} \
                --ppa-gain-rtol 1e-4"
            ;;
    esac
}


# ======================================================================
# Runner with skip-if-completed + 2-level OOM retry.
# ======================================================================
run_algo() {
    local ALGO=$1
    local OUT_DIR="${RESULTS_DIR}/${ALGO}/seed_${SEED}"
    local LOG_FILE="${LOG_DIR}/${ALGO}_seed${SEED}.log"
    local TIME_FILE="${TIME_DIR}/${ALGO}_seed${SEED}.time.txt"
    local DONE="${OUT_DIR}/.completed"
    local CSV="${OUT_DIR}/epoch_log.csv"

    if [ -f "$DONE" ]; then
        echo "[SKIP] ${ALGO} seed=${SEED} — already completed"
        return 0
    fi

    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"

    local BATCH_OVR=$COMMON_BATCH
    local KEFF=$K

    local STARTED_UTC T0 ENDED_UTC T1 ELAPSED STATUS
    local ATTEMPT=0
    local MAX_ATTEMPTS=3

    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT + 1))

        echo ""
        echo "----------------------------------------------------------------"
        echo "  [Attempt ${ATTEMPT}] ${ALGO}  seed=${SEED}  K=${KEFF}  batch=${BATCH_OVR}  GPUs=${NPROC}"
        echo "  log=${LOG_FILE}"
        echo "  Started: $(date)"
        echo "----------------------------------------------------------------"

        STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        T0=$(date +%s)

        local EXTRA_ARGS
        EXTRA_ARGS=$(algo_args "$ALGO" "$KEFF")

        local TIME_CMD=""
        if command -v /usr/bin/time >/dev/null 2>&1; then
            TIME_CMD="/usr/bin/time -v -o ${TIME_FILE}"
        fi

        # The shared --bb-* knobs are passed to every algorithm. They
        # are consumed only by methods using BB+Armijo (NPF, NN-DRO,
        # WRM, WFR, New_PPA); Madry/RO and Dual parse them and ignore
        # them because they use PGD and Langevin/MALA respectively.
        $TIME_CMD torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
            -m pretrained_input_icnn.main \
            --algorithm "$ALGO" \
            --pretrained-path "$PRETRAINED_PATH" \
            --data-dir "$DATA_DIR" \
            --epochs-adv "$EPOCHS_ADV" \
            --epochs-icnn-pretrain "$EPOCHS_ICNN_PRETRAIN" \
            --batch-size "$BATCH_OVR" \
            --num-workers "$NUM_WORKERS" \
            --lr-theta "$LR_THETA" \
            --penalty-lambda "$PENALTY_LAMBDA" \
            --inp-p "$INP_P" \
            --inp-eps "$INP_EPS" \
            --skip-pgd-during-train \
            --benchmark-mode \
            $([ "$USE_MARGIN_LOSS" = "1" ] && echo "--use-margin-loss") \
            "${BB_ARGS[@]}" \
            --seed "$SEED" \
            --save "${OUT_DIR}/final.pth" \
            --log-csv "$CSV" \
            $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"

        STATUS=${PIPESTATUS[0]}
        T1=$(date +%s)
        ENDED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        ELAPSED=$((T1 - T0))

        if [ $STATUS -eq 0 ]; then break; fi

        # OOM detection
        if grep -q -E "CUDA out of memory|OutOfMemoryError" "$LOG_FILE" 2>/dev/null; then
            local NEW_BATCH=$((BATCH_OVR / 2))
            if [ $NEW_BATCH -lt $((NPROC * 4)) ]; then
                echo "[FATAL OOM] ${ALGO}: cannot halve batch below ${NPROC}*4 = $((NPROC*4))"
                break
            fi
            echo "[OOM] retrying with batch_size ${BATCH_OVR} -> ${NEW_BATCH}"
            mv "$LOG_FILE" "${LOG_FILE}.attempt${ATTEMPT}"
            BATCH_OVR=$NEW_BATCH
            rm -rf "$OUT_DIR"
            mkdir -p "$OUT_DIR"
            continue
        fi

        echo "[FAIL] ${ALGO} seed=${SEED} non-OOM error (status=${STATUS}); see ${LOG_FILE}"
        break
    done

    if [ $STATUS -eq 0 ]; then
        touch "$DONE"
        printf "completed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$ALGO" "$SEED" "$NPROC" "$K" "$EPOCHS_ADV" \
            "$STARTED_UTC" "$ENDED_UTC" "$ELAPSED" \
            "$LOG_FILE" "$TIME_FILE" "$CSV" >> "$MANIFEST"
        echo "[OK] ${ALGO} seed=${SEED} elapsed=${ELAPSED}s"
    else
        printf "failed_%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$STATUS" "$ALGO" "$SEED" "$NPROC" "$K" "$EPOCHS_ADV" \
            "$STARTED_UTC" "$ENDED_UTC" "$ELAPSED" \
            "$LOG_FILE" "$TIME_FILE" "$CSV" >> "$MANIFEST"
    fi

    # Reset CUDA caches between algorithms (each is its own torchrun
    # process, so memory IS released — this is just diagnostic).
    sleep 2
}


# ======================================================================
# Split dispatcher
# ======================================================================
case $SPLIT in
    0)
        run_algo "npf"
        run_algo "nn_dro"
        run_algo "madry"
        run_algo "wrm"
        run_algo "wfr"
        run_algo "dual"
        run_algo "new_ppa"
        ;;
    1) run_algo "npf" ;;
    2) run_algo "nn_dro"; run_algo "madry" ;;
    3) run_algo "wrm";    run_algo "wfr"   ;;
    4) run_algo "dual";   run_algo "new_ppa" ;;
    *)
        echo "[FATAL] Unknown SPLIT=${SPLIT}. Use 0 (sequential) or 1-4."
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  Split ${SPLIT} complete at $(date)"
echo "================================================================"


# ======================================================================
# Aggregator: prints a clean comparison table + LaTeX, only when all 7
# algorithms in the current seed have finished.
# ======================================================================
SPLIT="$SPLIT" SEED="$SEED" RESULTS_DIR="$RESULTS_DIR" K="$K" \
NPROC="$NPROC" EPOCHS_ADV="$EPOCHS_ADV" python - << 'PYTHON_COLLECTOR'
import csv, json, os, statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ["RESULTS_DIR"])
SEED        = int(os.environ["SEED"])
K           = int(os.environ["K"])
NPROC       = int(os.environ["NPROC"])
EPOCHS_ADV  = int(os.environ["EPOCHS_ADV"])
ALGOS       = ["npf", "nn_dro", "madry", "wrm", "wfr", "dual", "new_ppa"]

manifest = RESULTS_DIR / "run_manifest.tsv"
done = set()
if manifest.exists():
    with manifest.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("status", "").startswith("completed") and int(r["seed"]) == SEED:
                done.add(r["algorithm"])

print(f"\n[COLLECTOR] {len(done)}/{len(ALGOS)} completed at SEED={SEED}, K={K}, GPUs={NPROC}")
if len(done) < len(ALGOS):
    print(f"[COLLECTOR] Missing: {sorted(set(ALGOS) - done)}")
    print("[COLLECTOR] Re-run after remaining splits complete for full table.")
    raise SystemExit(0)

def per_epoch_stats(csv_path: Path):
    if not csv_path.exists():
        return None
    epoch_secs, train_losses, train_accs = [], [], []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("phase", "adv") != "adv":
                continue
            try:
                ep = int(r["epoch"]); t = float(r["epoch_seconds"])
            except (KeyError, ValueError, TypeError):
                continue
            if ep == 1:  # drop cuDNN autotune warmup epoch
                continue
            epoch_secs.append(t)
            try:    train_losses.append(float(r["train_loss"]))
            except: pass
            try:    train_accs.append(float(r["train_acc"]))
            except: pass
    if not epoch_secs:
        return None
    return {
        "median_ep_s": statistics.median(epoch_secs),
        "min_ep_s":    min(epoch_secs),
        "max_ep_s":    max(epoch_secs),
        "n_epochs":    len(epoch_secs),
        "final_loss":  train_losses[-1] if train_losses else None,
        "final_acc":   train_accs[-1]   if train_accs   else None,
    }

def parse_time_v(p: Path):
    if not p.exists(): return {}
    out = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.strip().split(":", 1)
            out[k.strip()] = v.strip()
    return out

results = {}
for a in ALGOS:
    csvp = RESULTS_DIR / a / f"seed_{SEED}" / "epoch_log.csv"
    timep = RESULTS_DIR / "time" / f"{a}_seed{SEED}.time.txt"
    s = per_epoch_stats(csvp) or {}
    t = parse_time_v(timep)
    results[a] = {
        **s,
        "wallclock":  t.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "—"),
        "max_rss_kb": t.get("Maximum resident set size (kbytes)", "—"),
    }

print()
print("=" * 110)
print(f"  Input-space Adversarial Training — Runtime  (CIFAR-10, K={K}, seed {SEED}, {NPROC}-GPU DDP, {EPOCHS_ADV} ep)")
print("=" * 110)
hdr = (f"{'Algorithm':<10}  {'med ep (s)':>12}  {'min ep (s)':>12}  "
       f"{'max ep (s)':>12}  {'n ep':>5}  {'wallclock':>12}  {'RSS (MB)':>10}  "
       f"{'final_loss':>11}  {'final_acc':>10}")
print(hdr); print("-" * len(hdr))
for a in ALGOS:
    r = results[a]
    med = f"{r.get('median_ep_s', float('nan')):.3f}" if r.get("median_ep_s") is not None else "—"
    mn  = f"{r.get('min_ep_s', float('nan')):.3f}"    if r.get("min_ep_s")    is not None else "—"
    mx  = f"{r.get('max_ep_s', float('nan')):.3f}"    if r.get("max_ep_s")    is not None else "—"
    n   = r.get("n_epochs", "—")
    wc  = r.get("wallclock", "—")
    try:    rss = f"{float(r['max_rss_kb'])/1024:.1f}"
    except: rss = "—"
    fl  = f"{r.get('final_loss', float('nan')):.3f}" if r.get("final_loss") is not None else "—"
    fa  = f"{100*r.get('final_acc'):.2f}%" if r.get("final_acc") is not None else "—"
    print(f"{a:<10}  {med:>12}  {mn:>12}  {mx:>12}  {n!s:>5}  {wc:>12}  {rss:>10}  {fl:>11}  {fa:>10}")

# LaTeX table
print()
print("% ===== LaTeX table (paper-ready) =====")
print(r"\begin{table}[t] \centering")
print(f"\\caption{{Per-epoch wallclock for input-space adversarial training on CIFAR-10 (K={K}, "
      f"{NPROC}-GPU DDP, seed {SEED}). Median over {EPOCHS_ADV - 1} steady-state epochs (epoch 1 dropped).}}")
print(r"\label{tab:input_icnn_runtime}")
print(r"\begin{tabular}{l rrr r}")
print(r"\toprule")
print(r"Algorithm & Median (s) & Min (s) & Max (s) & Wallclock \\")
print(r"\midrule")
for a in ALGOS:
    r = results[a]
    med = f"{r['median_ep_s']:.2f}" if r.get("median_ep_s") is not None else "—"
    mn  = f"{r['min_ep_s']:.2f}"    if r.get("min_ep_s")    is not None else "—"
    mx  = f"{r['max_ep_s']:.2f}"    if r.get("max_ep_s")    is not None else "—"
    print(f"{a} & {med} & {mn} & {mx} & {r.get('wallclock', '—')} \\\\")
print(r"\bottomrule \end{tabular} \end{table}")

(RESULTS_DIR / f"summary_seed{SEED}.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: {RESULTS_DIR / f'summary_seed{SEED}.json'}")
PYTHON_COLLECTOR
