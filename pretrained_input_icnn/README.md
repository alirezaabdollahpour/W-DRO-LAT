# pretrained_input_icnn

Input-space adversarial training for CIFAR-10 using a pretrained classifier and a learned input transport adversary.

## Critical semantic defaults

For runs that should preserve the legacy `pretrained_INPUT_icnn.py` training semantics, keep these controls enabled:

```bash
TRANSPORT_COST=normalized_mse
ATTACK_CLEAN_CORRECT_ONLY=1
RESET_PARAMETRIC_BB_EACH_BATCH=1
```

These are now package defaults, and `run_runtime_sweep_ddp.sh` forwards `TRANSPORT_COST=normalized_mse` as `--transport-cost normalized_mse`. The stable-policy diagnostic command below separately sets `LR_THETA=0.1`; newer low-LR ablations should encode their own learning rate in `RUN_NAME` and `OUTPUT_FOLDER_NAME`.

`TRANSPORT_COST=normalized_mse` computes the per-sample mean squared distance in normalized CIFAR coordinates:

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

The stable-policy diagnostic run mirrors the old single-file training policy by letting BatchNorm participate in classifier training:

```bash
FREEZE_BATCHNORM=0
FREEZE_BATCHNORM_AFFINE=0
BATCHNORM_ONLINE_REFRESH=0
RECALIBRATE_BATCHNORM=0
```

Treat frozen BatchNorm as a separate ablation, not as the stable-policy reproduction:

```bash
FREEZE_BATCHNORM=1
FREEZE_BATCHNORM_AFFINE=1
BATCHNORM_ONLINE_REFRESH=0
RECALIBRATE_BATCHNORM=0
```

Only enable online BN refresh for an explicit BatchNorm-statistics ablation:

```bash
BATCHNORM_ONLINE_REFRESH=1
BATCHNORM_ONLINE_REFRESH_MOMENTUM=0.001
```

## Stable-policy DDP reproducibility run

This command records the current stable-policy NPF-LastQuad diagnostic run. It uses the corrected refactored code path, legacy normalized-MSE transport cost, clean-correct attack masking, per-batch BB reset, no BatchNorm freezing, and CE-based input PGD evaluation.

The run name and output folder use `lr0p1` because `LR_THETA=0.1`. Avoid naming this run `lr0p003`; that makes later artifact analysis ambiguous.

| Group | Setting | Value | Notes |
| --- | --- | --- | --- |
| Launcher | `SPLIT` | `0` | Runs `npf_lastquad`. |
| Launcher | `SEED` | `1` | Passed as positional arg 2. |
| Launcher | `NPROC` | `2` | Two GPUs via DDP. |
| Launcher | `K` / `OMEGA_STEPS` | `7` | Seven NPF adversary updates per batch. |
| Launcher | adversarial epochs | `50` | Positional arg 5 to `run_runtime_sweep_ddp.sh`. |
| Classifier | `LR_THETA` | `0.1` | SGD learning rate for ResNet-18. |
| Classifier | `COMMON_BATCH` | `512` | Global batch, split across ranks. |
| DRO | `PENALTY_LAMBDA` | `30` | Inner penalty multiplier. |
| DRO | `TRANSPORT_COST` | `normalized_mse` | Legacy normalized-coordinate mean squared transport cost. |
| DRO | `USE_MARGIN_LOSS` | `1` | Training adversary uses logsumexp-margin primary loss. |
| DRO | `LAMBDA_SCHEDULE` | empty | Fixed lambda. |
| DRO | `LAMBDA_STAGE_EPOCHS` | `0` | No staged schedule. |
| Legacy semantics | `ATTACK_CLEAN_CORRECT_ONLY` | `1` | Attack only clean-correct samples; clean-incorrect samples stay clean. |
| Legacy semantics | `RESET_PARAMETRIC_BB_EACH_BATCH` | `1` | Reset BB secant history each batch. |
| Warmup | `EPOCHS_ICNN_PRETRAIN` | `0` | No adversary-only warmup in this run. |
| BatchNorm | `FREEZE_BATCHNORM` | `0` | BatchNorm running stats update during classifier training. |
| BatchNorm | `FREEZE_BATCHNORM_AFFINE` | `0` | BN affine parameters remain trainable. |
| BatchNorm | `BATCHNORM_ONLINE_REFRESH` | `0` | No clean BN refresh pass. |
| BatchNorm | `RECALIBRATE_BATCHNORM` | `0` | No post-epoch BN recalibration. |
| NPF architecture | `NPF_LASTQUAD_HIDDEN` | `1024 512 512 256` | LastQuad hidden widths. |
| NPF architecture | `NPF_LASTQUAD_ACTIVATION` | `softplus` | Convex nonlinearity. |
| NPF architecture | `NPF_LASTQUAD_SOFTPLUS_BETA` | `20.0` | Matches the legacy ICNN softplus beta. |
| NPF architecture | `NPF_LASTQUAD_INIT_EPS` | `1e-4` | Identity-adjacent init scale. |
| NPF architecture | `NPF_LASTQUAD_STRONG_CONVEXITY` | `1.0` | Fixed strong-convexity term. |
| NPF optimizer | `NPF_INNER_OPTIMIZER` | `bb_armijo` | Custom BB+Armijo gradient ascent on NPF weights. |
| NPF optimizer | `BB_ALPHA0` | `5e-4` | Initial BB/Armijo step size. |
| NPF optimizer | `BB_ALPHA_MIN` | `1e-6` | Minimum step size. |
| NPF optimizer | `BB_ALPHA_MAX` | `1.0` | Maximum step size. |
| NPF optimizer | `BB_LS_C` | `0.1` | Armijo sufficient-increase constant. |
| NPF optimizer | `BB_LS_SHRINK` | `0.5` | Backtracking shrink factor. |
| NPF optimizer | `BB_LS_MAX_STEPS` | `10` | Maximum line-search trials. |
| PGD eval | `EVAL_PGD_SAMPLES` | `2000` | Test samples used for per-epoch input PGD. |
| PGD eval | `INPUT_PGD_LOSS` | `ce` | Cross-entropy PGD evaluation. |
| PGD eval | `INP_STEPS` | `20` | PGD steps per restart. |
| PGD eval | `INP_RESTARTS` | `5` | PGD random restarts. |
| Frozen map | `FROZEN_ADVERSARY_EPOCHS` | `0` | Disabled. |
| Frozen map | `FROZEN_ADVERSARY_MAP_STEPS` | `1` | Explicit safe default; unused when frozen epochs are zero. |

```bash
python csub.py -n lastquad-lam30-logsum-bb-k7-lr0p1-stablepolicy -g 2 -t 1d --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/LAT && \
    source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    RUN_NAME=npf_lq_lam30_logsum_lr0p1_K7_bb_stablepolicy_pgdce_seed1 \
    LR_THETA=0.1 \
    PENALTY_LAMBDA=30 \
    TRANSPORT_COST=normalized_mse \
    EPOCHS_ICNN_PRETRAIN=0 \
    LAMBDA_SCHEDULE='' \
    LAMBDA_STAGE_EPOCHS=0 \
    ATTACK_CLEAN_CORRECT_ONLY=1 \
    RESET_PARAMETRIC_BB_EACH_BATCH=1 \
    FREEZE_BATCHNORM=0 \
    FREEZE_BATCHNORM_AFFINE=0 \
    BATCHNORM_ONLINE_REFRESH=0 \
    RECALIBRATE_BATCHNORM=0 \
    USE_MARGIN_LOSS=1 \
    COMMON_BATCH=512 \
    NPF_LASTQUAD_HIDDEN='1024 512 512 256' \
    NPF_LASTQUAD_ACTIVATION=softplus \
    NPF_LASTQUAD_SOFTPLUS_BETA=20.0 \
    NPF_LASTQUAD_INIT_EPS=1e-4 \
    NPF_LASTQUAD_STRONG_CONVEXITY=1.0 \
    NPF_INNER_OPTIMIZER=bb_armijo \
    BB_ALPHA0=5e-4 \
    BB_ALPHA_MIN=1e-6 \
    BB_ALPHA_MAX=1.0 \
    BB_LS_C=0.1 \
    BB_LS_SHRINK=0.5 \
    BB_LS_MAX_STEPS=10 \
    EVAL_PGD_SAMPLES=2000 \
    INPUT_PGD_LOSS=ce \
    INP_STEPS=20 \
    INP_RESTARTS=5 \
    OMEGA_STEPS=7 \
    FROZEN_ADVERSARY_EPOCHS=0 \
    FROZEN_ADVERSARY_MAP_STEPS=1 \
    OUTPUT_FOLDER_NAME=BB_lam30_legacyfix_logsum_lr0p1_K7_pgdce_seed1 \
    bash run_runtime_sweep_ddp.sh 0 1 2 7 50"
```

### Exact legacy-script comparison knobs

For a closer comparison to the `26036bb` shell script, change only these fields after the above baseline is reproducible:

```bash
EPOCHS_ICNN_PRETRAIN=2
OUTPUT_FOLDER_NAME=BB_lam30_legacyfix_logsum_lr0p1_K7_warm2_pgdce_seed1
RUN_NAME=npf_lq_lam30_logsum_lr0p1_K7_bb_stablepolicy_warm2_pgdce_seed1
bash run_runtime_sweep_ddp.sh 0 1 2 7 30
```

## Margin-PGD ablation

To rerun the same training but evaluate PGD with the margin objective, change only these fields:

```bash
INPUT_PGD_LOSS=margin
RUN_NAME=npf_lq_lam30_logsum_lr0p1_K7_bb_stablepolicy_pgdmargin_seed1
OUTPUT_FOLDER_NAME=BB_lam30_legacyfix_logsum_lr0p1_K7_pgdmargin_seed1
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
