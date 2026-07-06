# HalfCheetah WDRO SAC

This directory contains a self-contained SAC experiment for the setup in the note:

- `nominal`: standard SAC.
- `pgd`: solves the Sinha-Duchi inner maximization over replay-buffer next states with PGD.
- `icnn`: learns an amortized ICNN/Brenier map `T(s') = grad psi(s')` using the NPF-LastQuad potential.

The robust target is:

```text
s_adv = arg sup_z {-V(z) - lambda ||z - s'||_M^2}
y = r + gamma V_target(s_adv)
```

where `V(s) = min(Q1(s, a), Q2(s, a)) - alpha log pi(a|s)`.

## Run

```bash
cd /mloscratch/homes/aabdolla/LAT/RL-HalfCheetah-v3
bash run_wdro_sac_halfcheetah.sh \
  --method icnn \
  --run-name halfcheetah_icnn_lam30 \
  --total-steps 1000000 \
  --eval-interval 10000 \
  --wandb
```

The script accepts the same environment-variable style as the CIFAR run:

```bash
PENALTY_LAMBDA=30 \
COMMON_BATCH=512 \
RUN_ONLY_ALGO=npf_lastquad \
NPF_LASTQUAD_HIDDEN='128 128 64 32' \
NPF_LASTQUAD_ACTIVATION=softplus \
NPF_LASTQUAD_SOFTPLUS_BETA=5.0 \
NPF_LASTQUAD_INIT_EPS=1e-4 \
NPF_LASTQUAD_STRONG_CONVEXITY=1.0 \
OMEGA_STEPS=20 \
INP_STEPS=20 \
INP_RESTARTS=5 \
BB_ALPHA0=1e-3 \
BB_ALPHA_MIN=1e-6 \
BB_ALPHA_MAX=0.05 \
BB_LS_C=1e-5 \
BB_LS_SHRINK=0.5 \
BB_LS_MAX_STEPS=20 \
bash run_wdro_sac_halfcheetah.sh
```

For baselines:

```bash
bash run_wdro_sac_halfcheetah.sh --method nominal --run-name halfcheetah_nominal
bash run_wdro_sac_halfcheetah.sh --method pgd --run-name halfcheetah_pgd_lam30
```

## Cluster Example

```bash
python csub.py -n hc-icnn-lam30 -g 1 -t 1d --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/LAT && \
    . /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    env \
    ENV_ID=HalfCheetah-v5 \
    RUN_NAME=halfcheetah_icnn_lam30 \
    RESULTS_DIR=/mloscratch/homes/aabdolla/LAT/RL-HalfCheetah-v3/runs \
    RUN_ONLY_ALGO=npf_lastquad \
    PENALTY_LAMBDA=30 \
    COMMON_BATCH=512 \
    NPF_LASTQUAD_HIDDEN='128 128 64 32' \
    NPF_LASTQUAD_ACTIVATION=softplus \
    NPF_LASTQUAD_SOFTPLUS_BETA=5.0 \
    NPF_LASTQUAD_INIT_EPS=1e-4 \
    NPF_LASTQUAD_STRONG_CONVEXITY=1.0 \
    OMEGA_STEPS=20 \
    INP_STEPS=20 \
    INP_RESTARTS=5 \
    BB_ALPHA0=1e-3 \
    BB_ALPHA_MIN=1e-6 \
    BB_ALPHA_MAX=0.05 \
    BB_LS_C=1e-5 \
    BB_LS_SHRINK=0.5 \
    BB_LS_MAX_STEPS=20 \
    bash RL-HalfCheetah-v3/run_wdro_sac_halfcheetah.sh \
      --total-steps 1000000 \
      --eval-interval 10000 \
      --wandb"
```

The trainer writes `args.json`, `metrics.jsonl`, and checkpoints under `runs/<run-name>/`. Robust evaluation logs return under mass, friction, and optional damping multipliers.
