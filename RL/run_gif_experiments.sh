#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

echo "=============================================="
echo " RL Agent Behavior GIF Experiments"
echo "=============================================="
echo ""
echo "GIFs will be saved to: ${SCRIPT_DIR}/gifs/"
echo ""

# -----------------------------------------------
# 1) CartPole — ICNN method
# -----------------------------------------------
echo ">>> [1/4] CartPole + ICNN"
echo "    params: env=cartpole method=icnn seed=0 iters=200 lam=2.0"
"${PYTHON_BIN}" RL_minimal.py \
  --method icnn \
  --env cartpole \
  --seed 0 \
  --iters 200 \
  --lam 2.0 \
  --k-icnn 10 \
  --eta-icnn 0.05 \
  --icnn-hidden-sizes 1024 512 512 256 128 64 \
  --icnn-init identity \
  --icnn-nonneg-init principled \
  --icnn-softplus-beta 20.0 \
  --save-gif \
  --gif-every 10 \
  --gif-fps 20 \
  --no-plot \
  "$@"
echo "    done."
echo ""

# -----------------------------------------------
# 2) CartPole — Particle method
# -----------------------------------------------
echo ">>> [2/4] CartPole + Particle"
echo "    params: env=cartpole method=particle seed=0 iters=200 lam=2.0"
"${PYTHON_BIN}" RL_minimal.py \
  --method particle \
  --env cartpole \
  --seed 0 \
  --iters 200 \
  --lam 2.0 \
  --save-gif \
  --gif-every 10 \
  --gif-fps 20 \
  --no-plot \
  "$@"
echo "    done."
echo ""

# -----------------------------------------------
# 3) Swing-Up Pendulum — ICNN method
# -----------------------------------------------
echo ">>> [3/4] SwingUp Pendulum + ICNN"
echo "    params: env=swingup_pendulum method=icnn seed=0 iters=200 lam=2.0 softplus_beta=20.0"
"${PYTHON_BIN}" RL_minimal.py \
  --method icnn \
  --env swingup_pendulum \
  --seed 0 \
  --iters 200 \
  --lam 2.0 \
  --k-icnn 10 \
  --eta-icnn 0.05 \
  --icnn-hidden-sizes 1024 512 512 256 128 64 \
  --icnn-init identity \
  --icnn-nonneg-init principled \
  --icnn-softplus-beta 20.0 \
  --save-gif \
  --gif-every 10 \
  --gif-fps 20 \
  --no-plot \
  "$@"
echo "    done."
echo ""

# -----------------------------------------------
# 4) Swing-Up Pendulum — Particle method
# -----------------------------------------------
echo ">>> [4/4] SwingUp Pendulum + Particle"
echo "    params: env=swingup_pendulum method=particle seed=0 iters=200 lam=2.0"
"${PYTHON_BIN}" RL_minimal.py \
  --method particle \
  --env swingup_pendulum \
  --seed 0 \
  --iters 200 \
  --lam 2.0 \
  --save-gif \
  --gif-every 10 \
  --gif-fps 20 \
  --no-plot \
  "$@"
echo "    done."
echo ""

echo "=============================================="
echo " All experiments complete!"
echo " GIFs saved in: ${SCRIPT_DIR}/gifs/"
echo "=============================================="
echo ""
echo "Output directories:"
ls -d "${SCRIPT_DIR}/gifs/"*/ 2>/dev/null || echo "  (no gifs yet)"
