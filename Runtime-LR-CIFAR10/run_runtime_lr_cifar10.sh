#!/usr/bin/env bash
set -Eeuo pipefail

# Runtime inner-iteration sweep for Logistic_Regression_CIFAR10/CIFAR10_LogReg.py.
#
# Defaults run each method separately so wall-clock times are method-specific:
#   1. Implicit-map baselines: WRM Dual WFR RO
#   2. ICNN-DRO: NPF
#
# WFR is invoked through the CIFAR-10 method registry as "WFR". The
# Least_Squares/algorithms/wfr.py file is a separate least-squares
# implementation and is not directly callable by CIFAR10_LogReg.py.
#
# Results are written under Runtime-LR-CIFAR10/results by default.
#
# Useful overrides:
#   SEEDS="2022 2023 2024" bash Runtime-LR-CIFAR10/run_runtime_lr_cifar10.sh
#   K_VALUES="5 10 25" bash Runtime-LR-CIFAR10/run_runtime_lr_cifar10.sh
#   OVERWRITE=1 bash Runtime-LR-CIFAR10/run_runtime_lr_cifar10.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CIFAR_DIR="${REPO_ROOT}/Logistic_Regression_CIFAR10"
PYTHON_BIN="${PYTHON_BIN:-python}"

RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results}"
LOG_DIR="${RESULT_ROOT}/logs"
TIME_DIR="${RESULT_ROOT}/time"
MANIFEST="${RESULT_ROOT}/run_manifest.tsv"
CANONICAL_FEATURE_CACHE="${CIFAR_DIR}/data/cifar10_resnet50_features.pth"
if [[ -n "${FEATURE_CACHE:-}" ]]; then
  FEATURE_CACHE="${FEATURE_CACHE}"
elif [[ -f "${CANONICAL_FEATURE_CACHE}" ]]; then
  FEATURE_CACHE="${CANONICAL_FEATURE_CACHE}"
else
  FEATURE_CACHE="${RESULT_ROOT}/cache/cifar10_resnet50_features.pth"
fi
DATA_DIR="${DATA_DIR:-${CIFAR_DIR}/data}"

K_VALUES="${K_VALUES:-20}"
SEEDS="${SEEDS:-2022}"
EPOCHS="${EPOCHS:-10}"
EPS_ENT="${EPS_ENT:-0.2}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PGD_STEPS="${PGD_STEPS:-20}"
PGD_RESTARTS="${PGD_RESTARTS:-5}"
RO_EPSILON_LEVEL="${RO_EPSILON_LEVEL:-0.04}"
NPF_HIDDEN="${NPF_HIDDEN:-512 512 256 128 64}"
NPF_OUTER_RANK="${NPF_OUTER_RANK:-16}"
NPF_INNER_RANK="${NPF_INNER_RANK:-2}"
NPF_ACTIVATION="${NPF_ACTIVATION:-softplus}"
NPF_ELU_ALPHA="${NPF_ELU_ALPHA:-1.0}"
NPF_SOFTPLUS_BETA="${NPF_SOFTPLUS_BETA:-10.0}"
NPF_INIT_EPS="${NPF_INIT_EPS:-1e-4}"
NPF_LR_B="${NPF_LR_B:-1e-3}"
NPF_WEIGHT_DECAY_B="${NPF_WEIGHT_DECAY_B:-1e-4}"
NPF_BB_ALPHA0="${NPF_BB_ALPHA0:-2e-4}"
NPF_BB_ALPHA_MIN="${NPF_BB_ALPHA_MIN:-1e-7}"
NPF_BB_ALPHA_MAX="${NPF_BB_ALPHA_MAX:-0.25}"
NPF_BB_LS_C="${NPF_BB_LS_C:-1e-4}"
NPF_BB_LS_SHRINK="${NPF_BB_LS_SHRINK:-0.5}"
NPF_BB_LS_MAX_STEPS="${NPF_BB_LS_MAX_STEPS:-15}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OVERWRITE="${OVERWRITE:-0}"

WRM_STYLE_METHODS=(WRM Dual WFR RO)
ICNN_STYLE_METHODS=(NPF)

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${TIME_DIR}" "$(dirname "${FEATURE_CACHE}")"

if [[ ! -f "${CIFAR_DIR}/CIFAR10_LogReg.py" ]]; then
  echo "ERROR: Could not find ${CIFAR_DIR}/CIFAR10_LogReg.py" >&2
  exit 1
fi

{
  echo "# Runtime LR CIFAR-10 inner-iteration sweep"
  echo "# started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# repo_root=${REPO_ROOT}"
  echo "# cifar_dir=${CIFAR_DIR}"
  echo "# python=$(${PYTHON_BIN} --version 2>&1)"
  echo "# k_values=${K_VALUES}"
  echo "# seeds=${SEEDS}"
  echo "# epochs=${EPOCHS}"
  echo "# eps_ent=${EPS_ENT}"
  echo "# num_samples=${NUM_SAMPLES}"
  echo "# batch_size=${BATCH_SIZE}"
  echo "# pgd_steps=${PGD_STEPS}"
  echo "# pgd_restarts=${PGD_RESTARTS}"
  echo "# ro_epsilon_level=${RO_EPSILON_LEVEL}"
  echo "# npf_hidden=${NPF_HIDDEN}"
  echo "# npf_outer_rank=${NPF_OUTER_RANK}"
  echo "# npf_inner_rank=${NPF_INNER_RANK}"
  echo "# npf_activation=${NPF_ACTIVATION}"
  echo "# npf_softplus_beta=${NPF_SOFTPLUS_BETA}"
  echo "# npf_init_eps=${NPF_INIT_EPS}"
  echo "# npf_lr_B=${NPF_LR_B}"
  echo "# npf_weight_decay_B=${NPF_WEIGHT_DECAY_B}"
  echo "# npf_bb_alpha0=${NPF_BB_ALPHA0}"
  echo "# npf_bb_alpha_min=${NPF_BB_ALPHA_MIN}"
  echo "# npf_bb_alpha_max=${NPF_BB_ALPHA_MAX}"
  echo "# npf_bb_ls_c=${NPF_BB_LS_C}"
  echo "# npf_bb_ls_shrink=${NPF_BB_LS_SHRINK}"
  echo "# npf_bb_ls_max_steps=${NPF_BB_LS_MAX_STEPS}"
  echo "# feature_cache=${FEATURE_CACHE}"
  echo "# data_dir=${DATA_DIR}"
  echo -e "status\tfamily\tk\tseed\tstarted_utc\tended_utc\telapsed_seconds\tout_dir\tlog_file\ttime_file\tcommand"
} > "${MANIFEST}.tmp"

if [[ -f "${MANIFEST}" ]]; then
  cat "${MANIFEST}" >> "${MANIFEST}.previous"
fi
mv "${MANIFEST}.tmp" "${MANIFEST}"

join_by_space() {
  local first=1
  local item
  for item in "$@"; do
    if [[ "${first}" -eq 1 ]]; then
      printf "%s" "${item}"
      first=0
    else
      printf " %s" "${item}"
    fi
  done
}

run_one() {
  local family="$1"
  local k="$2"
  local seed="$3"
  shift 3
  local -a methods=("$@")
  local -a npf_hidden_args=()

  local method_tag
  method_tag="$(join_by_space "${methods[@]}")"
  method_tag="${method_tag// /-}"

  local out_dir="${RESULT_ROOT}/${family}/K_${k}/seed_${seed}"
  local log_file="${LOG_DIR}/${family}_K${k}_seed${seed}.log"
  local time_file="${TIME_DIR}/${family}_K${k}_seed${seed}.time.txt"
  local done_file="${out_dir}/.completed"

  if [[ "${OVERWRITE}" != "1" && -f "${done_file}" ]]; then
    echo "SKIP completed: family=${family} K=${k} seed=${seed}"
    printf "skipped\t%s\t%s\t%s\t\t\t\t%s\t%s\t%s\t%s\n" \
      "${family}" "${k}" "${seed}" "${out_dir}" "${log_file}" "${time_file}" "already_completed" \
      >> "${MANIFEST}"
    return 0
  fi

  if [[ "${OVERWRITE}" == "1" ]]; then
    rm -rf "${out_dir}"
  fi
  mkdir -p "${out_dir}"

  local -a cmd=(
    "${PYTHON_BIN}" CIFAR10_LogReg.py
    --methods "${methods[@]}"
    --epochs "${EPOCHS}"
    --eps_ent "${EPS_ENT}"
    --num_samples "${NUM_SAMPLES}"
    --batch_size "${BATCH_SIZE}"
    --pgd_steps "${PGD_STEPS}"
    --pgd_restarts "${PGD_RESTARTS}"
    --ro_epsilon_level "${RO_EPSILON_LEVEL}"
    --num_workers "${NUM_WORKERS}"
    --seed "${seed}"
    --out_dir "${out_dir}"
    --feature_cache "${FEATURE_CACHE}"
    --data_dir "${DATA_DIR}"
  )

  if [[ "${family}" == wrm_style* ]]; then
    cmd+=(--inner_steps "${k}" --ro_pgd_steps "${k}")
  elif [[ "${family}" == icnn_style* ]]; then
    read -r -a npf_hidden_args <<< "${NPF_HIDDEN}"
    cmd+=(
      --icnn_omega_steps_per_batch "${k}"
      --inner_steps_npf "${k}"
      --nn_dro_omega_steps_per_batch "${k}"
      --npf_hidden "${npf_hidden_args[@]}"
      --npf_outer_rank "${NPF_OUTER_RANK}"
      --npf_inner_rank "${NPF_INNER_RANK}"
      --npf_activation "${NPF_ACTIVATION}"
      --npf_elu_alpha "${NPF_ELU_ALPHA}"
      --npf_softplus_beta "${NPF_SOFTPLUS_BETA}"
      --npf_init_eps "${NPF_INIT_EPS}"
      --npf_lr_B "${NPF_LR_B}"
      --npf_weight_decay_B "${NPF_WEIGHT_DECAY_B}"
      --npf_bb_alpha0 "${NPF_BB_ALPHA0}"
      --npf_bb_alpha_min "${NPF_BB_ALPHA_MIN}"
      --npf_bb_alpha_max "${NPF_BB_ALPHA_MAX}"
      --npf_bb_ls_c "${NPF_BB_LS_C}"
      --npf_bb_ls_shrink "${NPF_BB_LS_SHRINK}"
      --npf_bb_ls_max_steps "${NPF_BB_LS_MAX_STEPS}"
    )
  else
    echo "ERROR: Unknown family '${family}'" >&2
    exit 1
  fi

  local started_utc ended_utc t0 t1 elapsed status
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0="$(date +%s)"

  echo "RUN family=${family} methods=${method_tag} K=${k} seed=${seed}"
  echo "  out_dir=${out_dir}"
  echo "  log=${log_file}"
  echo "  time=${time_file}"

  set +e
  (
    cd "${CIFAR_DIR}"
    export PYTHONUNBUFFERED=1
    if command -v /usr/bin/time >/dev/null 2>&1; then
      /usr/bin/time -v -o "${time_file}" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  ) >"${log_file}" 2>&1
  status=$?
  set -e

  t1="$(date +%s)"
  ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed="$((t1 - t0))"

  local command_string
  printf -v command_string "%q " "${cmd[@]}"
  command_string="${command_string% }"

  if [[ "${status}" -eq 0 ]]; then
    touch "${done_file}"
    printf "completed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${family}" "${k}" "${seed}" "${started_utc}" "${ended_utc}" "${elapsed}" \
      "${out_dir}" "${log_file}" "${time_file}" "${command_string}" >> "${MANIFEST}"
    echo "DONE family=${family} K=${k} seed=${seed} elapsed=${elapsed}s"
  else
    printf "failed_%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${status}" "${family}" "${k}" "${seed}" "${started_utc}" "${ended_utc}" \
      "${elapsed}" "${out_dir}" "${log_file}" "${time_file}" "${command_string}" >> "${MANIFEST}"
    echo "FAILED family=${family} K=${k} seed=${seed}; see ${log_file}" >&2
    exit "${status}"
  fi
}

for seed in ${SEEDS}; do
  for k in ${K_VALUES}; do
    # for method in "${WRM_STYLE_METHODS[@]}"; do
    #   run_one "wrm_style_${method}" "${k}" "${seed}" "${method}"
    # done
    for method in "${ICNN_STYLE_METHODS[@]}"; do
      run_one "icnn_style_${method}" "${k}" "${seed}" "${method}"
    done
  done
done

echo "All requested runs finished. Manifest: ${MANIFEST}"
