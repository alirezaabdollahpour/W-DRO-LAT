#!/usr/bin/env bash
#
# Fair runtime comparison across all 7 algorithms on a single GPU.
#
# Modelled after Runtime-LR-CIFAR10/run_runtime_lr_cifar10.sh:
#   * Sequential, never concurrent — one algorithm at a time.
#   * /usr/bin/time -v records wallclock + max RSS + CPU% for each run.
#   * RESULT_ROOT/run_manifest.tsv tracks every (algorithm, seed) cell.
#
# Defaults are tuned so each algorithm gets a comparable inner-loop
# budget (K=10 inner iterations) AND the per-epoch PGD evaluation is
# disabled — eval cost would otherwise dominate short runs and bias
# the comparison toward eval throughput rather than training cost.
#
# Tips for clean numbers:
#   * Make sure no other GPU process is running (nvidia-smi).
#   * Set CUDA_VISIBLE_DEVICES to a single device.
#   * Pin CPU threads if dataloader contention varies:
#       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash run_runtime_sweep.sh
#   * The first epoch is always slower (cuDNN autotune warm-up). Compare
#     median per-epoch time from epoch 2 onwards using the CSV log.
#
# Useful overrides:
#   ALGORITHMS="npf madry"          subset of algorithms to run
#   SEEDS="1 2 3"                   repeat each algo with different seeds
#   EPOCHS_ADV=10                   shorter / longer runs
#   K=10                            shared inner-iteration budget
#   PRETRAINED_PATH=/path/to/R2.pth
#   OVERWRITE=1                     redo runs marked completed

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ALGORITHMS="${ALGORITHMS:-npf nn_dro madry wrm wfr dual new_ppa}"
SEEDS="${SEEDS:-1}"
EPOCHS_ADV="${EPOCHS_ADV:-30}"
EPOCHS_ICNN_PRETRAIN="${EPOCHS_ICNN_PRETRAIN:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PENALTY_LAMBDA="${PENALTY_LAMBDA:-30}"
LR_THETA="${LR_THETA:-0.1}"
K="${K:-10}"
INP_P="${INP_P:-2}"
INP_EPS="${INP_EPS:-0.5}"
USE_MARGIN_LOSS="${USE_MARGIN_LOSS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints/R2.pth}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"

RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/runtime_sweep_results}"
LOG_DIR="${RESULT_ROOT}/logs"
TIME_DIR="${RESULT_ROOT}/time"
MANIFEST="${RESULT_ROOT}/run_manifest.tsv"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${TIME_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  printf "status\talgorithm\tseed\tstarted_utc\tended_utc\telapsed_seconds\tepochs\tk\tlog_file\ttime_file\tcsv\n" > "${MANIFEST}"
fi

# Fairness rationale per algorithm:
#   npf, nn_dro      — both use --omega-steps-per-batch K (parametric ascent)
#   wrm              — --wrm-inner-steps K (per-batch ascent steps)
#   wfr              — --wfr-inner-steps K (per-batch sampler steps)
#   madry            — --madry-pgd-steps K (PGD ascent steps)
#   new_ppa          — --ppa-round0-steps K, --ppa-refine-steps K
#                       (the WRM-style ascent step counts)
#   dual             — has no per-batch inner ascent (m sampled per batch
#                       from --dual-sample-level); K controls sample_level
#                       so 2^sample_level scales ≈ same as the others.
algo_specific_args() {
  local algo="$1"
  case "${algo}" in
    npf)
      echo "--omega-steps-per-batch ${K} \
            --npf-hidden 1024 512 512 256 128 64 \
            --npf-outer-rank 8 --npf-inner-rank 2 \
            --npf-activation softplus --npf-softplus-beta 10.0 \
            --npf-init-eps 1e-4 --npf-strong-convexity 1.0 \
            --npf-bb-alpha0 2e-4 --npf-bb-alpha-min 1e-7 --npf-bb-alpha-max 0.25 \
            --npf-bb-ls-c 1e-4 --npf-bb-ls-shrink 0.5 --npf-bb-ls-max-steps 15"
      ;;
    nn_dro)
      echo "--omega-steps-per-batch ${K} \
            --nn-dro-hidden 512 512 256 256 128 \
            --nn-dro-activation relu --nn-dro-softplus-beta 20.0 \
            --nn-dro-init-scale 1e-3 --nn-dro-inner-lr 1e-2"
      ;;
    madry)
      echo "--madry-epsilon ${INP_EPS} --madry-pgd-steps ${K} --madry-pgd-restarts 1"
      ;;
    wrm)
      echo "--wrm-inner-steps ${K} --wrm-inner-lr 1e-2"
      ;;
    wfr)
      echo "--wfr-inner-steps ${K} --wfr-num-samples 8 --wfr-epsilon 0.1 --wfr-inner-lr 1e-2"
      ;;
    dual)
      # 2^K can blow up memory at K=10 (m=1024); cap sample_level at 5.
      local lvl="${K}"
      if (( lvl > 5 )); then lvl=5; fi
      echo "--dual-epsilon 1e-3 --dual-sample-level ${lvl}"
      ;;
    new_ppa)
      echo "--ppa-num-rounds 5 --ppa-min-rounds 2 \
            --ppa-round0-steps ${K} --ppa-round0-lr 1e-2 \
            --ppa-refine-steps ${K} --ppa-refine-lr 5e-3 \
            --ppa-gain-rtol 1e-4"
      ;;
    *)
      echo "ERROR: unknown algorithm ${algo}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local algo="$1" seed="$2"
  local out_dir="${RESULT_ROOT}/${algo}/seed_${seed}"
  local log_file="${LOG_DIR}/${algo}_seed${seed}.log"
  local time_file="${TIME_DIR}/${algo}_seed${seed}.time.txt"
  local csv="${out_dir}/epoch_log.csv"
  local done_file="${out_dir}/.completed"

  if [[ "${OVERWRITE}" != "1" && -f "${done_file}" ]]; then
    echo "SKIP completed: algo=${algo} seed=${seed}"
    return 0
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then rm -rf "${out_dir}"; fi
  mkdir -p "${out_dir}"

  local extra_args
  extra_args="$(algo_specific_args "${algo}")"

  local -a base_args=(
    --algorithm "${algo}"
    --pretrained-path "${PRETRAINED_PATH}"
    --data-dir "${DATA_DIR}"
    --epochs-adv "${EPOCHS_ADV}"
    --epochs-icnn-pretrain "${EPOCHS_ICNN_PRETRAIN}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lr-theta "${LR_THETA}"
    --penalty-lambda "${PENALTY_LAMBDA}"
    --inp-p "${INP_P}"
    --inp-eps "${INP_EPS}"
    --skip-pgd-during-train
    --benchmark-mode
    --seed "${seed}"
    --save "${out_dir}/final.pth"
    --log-csv "${csv}"
  )
  if [[ "${USE_MARGIN_LOSS}" == "1" ]]; then
    base_args+=(--use-margin-loss)
  fi

  local started_utc t0 t1 ended_utc elapsed status
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0="$(date +%s)"
  echo ">> RUN algo=${algo} seed=${seed} K=${K} epochs=${EPOCHS_ADV}"
  echo "   log=${log_file}"

  set +e
  (
    cd "${SCRIPT_DIR}"
    export PYTHONUNBUFFERED=1
    if command -v /usr/bin/time >/dev/null 2>&1; then
      /usr/bin/time -v -o "${time_file}" \
        "${PYTHON_BIN}" -m pretrained_input_icnn.main \
          "${base_args[@]}" ${extra_args}
    else
      "${PYTHON_BIN}" -m pretrained_input_icnn.main \
        "${base_args[@]}" ${extra_args}
    fi
  ) >"${log_file}" 2>&1
  status=$?
  set -e

  t1="$(date +%s)"
  ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed="$((t1 - t0))"

  if [[ "${status}" -eq 0 ]]; then
    touch "${done_file}"
    printf "completed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${algo}" "${seed}" "${started_utc}" "${ended_utc}" "${elapsed}" \
      "${EPOCHS_ADV}" "${K}" "${log_file}" "${time_file}" "${csv}" \
      >> "${MANIFEST}"
    echo "<< DONE algo=${algo} seed=${seed} elapsed=${elapsed}s"
  else
    printf "failed_%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${status}" "${algo}" "${seed}" "${started_utc}" "${ended_utc}" "${elapsed}" \
      "${EPOCHS_ADV}" "${K}" "${log_file}" "${time_file}" "${csv}" \
      >> "${MANIFEST}"
    echo "<< FAILED algo=${algo} seed=${seed} (see ${log_file})" >&2
    exit "${status}"
  fi
}

# Sanity probe so we fail fast if the GPU is contended.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi pre-check:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo
fi

for seed in ${SEEDS}; do
  for algo in ${ALGORITHMS}; do
    run_one "${algo}" "${seed}"
  done
done

echo "All runs done. Manifest: ${MANIFEST}"
echo "Per-epoch CSV logs in: ${RESULT_ROOT}/<algo>/seed_<seed>/epoch_log.csv"
