# pretrained_input_icnn

Input-space adversarial training for CIFAR-10 using a pretrained classifier and a learned input transport adversary.

## Critical default

For runs that should match the legacy `pretrained_INPUT_icnn.py` implementation, use these defaults:

```bash
TRANSPORT_COST=normalized_mse
ATTACK_CLEAN_CORRECT_ONLY=1
RESET_PARAMETRIC_BB_EACH_BATCH=1
LR_THETA=0.003
```

This is now the package default, and `run_runtime_sweep_ddp.sh` forwards it as `--transport-cost normalized_mse`. It computes the per-sample mean squared distance in normalized CIFAR coordinates:

```python
((x_adv_norm - x_norm) ** 2).reshape(batch, -1).mean(dim=1)
```

Do not use `TRANSPORT_COST=pixel_l2_squared` unless you intentionally want the newer pixel-space squared-L2 ablation. With the same `PENALTY_LAMBDA`, that changes the scale of the inner DRO penalty substantially.

`ATTACK_CLEAN_CORRECT_ONLY=1` matches the legacy outer loop: the learned transport is trained and applied only on examples that the clean classifier currently gets right; clean-misclassified examples stay at the clean input for the classifier update.

`RESET_PARAMETRIC_BB_EACH_BATCH=1` matches the legacy BB+Armijo loop: the BB secant history is reset at every batch. Carrying BB history across batches is an ablation because the stochastic objective changes with both the batch and the classifier.

## PGD evaluation loss

`INPUT_PGD_LOSS` controls only the evaluation PGD attack objective.

```bash
INPUT_PGD_LOSS=ce      # cross-entropy PGD evaluation
INPUT_PGD_LOSS=margin  # logsumexp-margin PGD evaluation
```

`USE_MARGIN_LOSS=1` controls the training adversary objective. A run can train with the logsumexp-margin adversary objective while evaluating PGD with CE:

```bash
USE_MARGIN_LOSS=1
INPUT_PGD_LOSS=ce
```

## BatchNorm controls

The recommended baseline after the transport-cost fix is to keep the pretrained classifier BatchNorm fixed:

```bash
FREEZE_BATCHNORM=1
FREEZE_BATCHNORM_AFFINE=1
RECALIBRATE_BATCHNORM=0
```

Only enable online BN refresh for an explicit ablation:

```bash
BATCHNORM_ONLINE_REFRESH=1
BATCHNORM_ONLINE_REFRESH_MOMENTUM=0.001
```

## Standard DDP run through csub

This is the corrected K=15 NPF-LastQuad command on two H100 GPUs. It trains with logsumexp-margin adversary loss, evaluates PGD with CE, and uses the legacy normalized-MSE transport penalty.

```bash
python csub.py -n lastquad-lam30-logsum-bb-k15-legacycost-pgdce-bnonline -g 2 -t 1d --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/LAT && \
    source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    RUN_NAME=npf_lq_lam30_logsum_lr0p003_K15_bb_legacycost_bnonline_mom0p001_pgdce_seed1 \
    LR_THETA=0.003 \
    PENALTY_LAMBDA=30 \
    TRANSPORT_COST=normalized_mse \
    ATTACK_CLEAN_CORRECT_ONLY=1 \
    RESET_PARAMETRIC_BB_EACH_BATCH=1 \
    LAMBDA_SCHEDULE='' \
    LAMBDA_STAGE_EPOCHS=0 \
    FREEZE_BATCHNORM=1 \
    FREEZE_BATCHNORM_AFFINE=1 \
    BATCHNORM_ONLINE_REFRESH=1 \
    BATCHNORM_ONLINE_REFRESH_MOMENTUM=0.001 \
    RECALIBRATE_BATCHNORM=0 \
    USE_MARGIN_LOSS=1 \
    COMMON_BATCH=512 \
    NPF_LASTQUAD_HIDDEN='1024 512 512 256 128 64' \
    NPF_LASTQUAD_ACTIVATION=softplus \
    NPF_LASTQUAD_SOFTPLUS_BETA=10.0 \
    NPF_LASTQUAD_INIT_EPS=1e-4 \
    NPF_LASTQUAD_STRONG_CONVEXITY=1.0 \
    NPF_INNER_OPTIMIZER=bb_armijo \
    BB_ALPHA0=1e-3 \
    BB_ALPHA_MIN=1e-6 \
    BB_ALPHA_MAX=0.05 \
    BB_LS_C=1e-5 \
    BB_LS_SHRINK=0.5 \
    BB_LS_MAX_STEPS=20 \
    EVAL_PGD_SAMPLES=2000 \
    INPUT_PGD_LOSS=ce \
    INP_STEPS=20 \
    INP_RESTARTS=5 \
    OMEGA_STEPS=15 \
    FROZEN_ADVERSARY_EPOCHS=0 \
    FROZEN_ADVERSARY_MAP_STEPS=1 \
    OUTPUT_FOLDER_NAME=BB_lam30_legacycost_logsum_lr0p003_K15_bnonline_mom0p001_pgdce \
    bash run_runtime_sweep_ddp.sh 0 1 2 15 50"
```

Set `FROZEN_ADVERSARY_MAP_STEPS=1` even when `FROZEN_ADVERSARY_EPOCHS=0`. The config validator requires map steps to be at least 1.

## Margin-PGD ablation

To rerun the same training but evaluate PGD with the margin objective, change only these fields:

```bash
INPUT_PGD_LOSS=margin
RUN_NAME=npf_lq_lam30_logsum_lr0p003_K15_bb_legacycost_bnonline_mom0p001_pgdmargin_seed1
OUTPUT_FOLDER_NAME=BB_lam30_legacycost_logsum_lr0p003_K15_bnonline_mom0p001_pgdmargin
```

## Local smoke test

Before launching a long job, this one-batch smoke run verifies that the entrypoint, checkpoint loader, transport-cost flag, and PGD evaluator all parse correctly:

```bash
SMOKE_MAX_TRAIN_BATCHES=1 /mloscratch/homes/aabdolla/optiselect/.venv/bin/python -m pretrained_input_icnn.main \
  --algorithm npf_lastquad \
  --pretrained-path ResNet_checkpoints/R2.pth \
  --data-dir ./data \
  --epochs-adv 1 \
  --batch-size 4 \
  --num-workers 0 \
  --lr-theta 0.003 \
  --penalty-lambda 30 \
  --transport-cost normalized_mse \
  --attack-clean-correct-only \
  --reset-parametric-bb-each-batch \
  --omega-steps-per-batch 0 \
  --npf-lastquad-hidden 4 \
  --eval-input-pgd \
  --eval-input-pgd-samples 8 \
  --input-pgd-loss ce \
  --inp-p 2 \
  --inp-eps 0.5 \
  --inp-steps 1 \
  --inp-restarts 1 \
  --no-freeze-batchnorm \
  --no-freeze-batchnorm-affine \
  --no-online-batchnorm-refresh \
  --no-recalibrate-batchnorm \
  --save /tmp/input_icnn_smoke.pth \
  --log-csv /tmp/input_icnn_smoke.csv
```

## Validation commands

Run these before pushing changes:

```bash
python -m py_compile pretrained_input_icnn/config.py pretrained_input_icnn/main.py pretrained_input_icnn/utils/eval.py pretrained_input_icnn/utils/projections.py pretrained_input_icnn/algorithms/base.py pretrained_input_icnn/algorithms/npf.py pretrained_input_icnn/algorithms/nn_dro.py pretrained_input_icnn/algorithms/wrm.py pretrained_input_icnn/algorithms/wfr.py pretrained_input_icnn/algorithms/dual.py pretrained_input_icnn/algorithms/new_ppa.py
bash -n run_pretrained_input_icnn.sh run_runtime_sweep_ddp.sh
python -m pytest tests/test_pretrained_input_icnn_inner_steps.py
```
