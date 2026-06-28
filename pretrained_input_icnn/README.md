# pretrained_input_icnn

Input-space adversarial training for CIFAR-10 using a pretrained classifier and a learned input transport adversary.

## Critical semantic defaults

For runs that should preserve the legacy `pretrained_INPUT_icnn.py` training semantics, keep these controls enabled:

```bash
TRANSPORT_COST=normalized_mse
ATTACK_CLEAN_CORRECT_ONLY=1
RESET_PARAMETRIC_BB_EACH_BATCH=1
PARAMETRIC_BB_MAX_GRAD_NORM=1.0
```

These are now package defaults, and `run_runtime_sweep_ddp.sh` forwards `TRANSPORT_COST=normalized_mse` as `--transport-cost normalized_mse`. The stable-policy diagnostic command below separately sets `LR_THETA=0.1`; newer low-LR ablations should encode their own learning rate in `RUN_NAME` and `OUTPUT_FOLDER_NAME`.

`TRANSPORT_COST=normalized_mse` computes the per-sample mean squared distance in normalized CIFAR coordinates:

```python
((x_adv_norm - x_norm) ** 2).reshape(batch, -1).mean(dim=1)
```

Do not use `TRANSPORT_COST=pixel_l2_squared` unless you intentionally want the newer pixel-space squared-L2 ablation. With the same `PENALTY_LAMBDA`, that changes the scale of the inner DRO penalty substantially.

`ATTACK_CLEAN_CORRECT_ONLY=1` matches the legacy outer loop: the learned transport is trained and applied only on examples that the clean classifier currently gets right; clean-misclassified examples stay at the clean input for the classifier update.

`RESET_PARAMETRIC_BB_EACH_BATCH=1` matches the legacy BB+Armijo loop: the BB secant history is reset at every batch. Carrying BB history across batches is an ablation because the stochastic objective changes with both the batch and the classifier.

`PARAMETRIC_BB_MAX_GRAD_NORM=1.0` matches the legacy ICNN ascent loop: the shared parametric adversary gradient is clipped to global norm 1.0 before the BB step proposal and before Armijo trial steps. Set it to `0` only for a clipping ablation.

## NPF / BB+Armijo fixes

The current refactored NPF path includes the following fixes relative to the earlier `pretrained_input_icnn` implementation:

| Area | Fix | Why it matters |
| --- | --- | --- |
| NPF LastQuad initialization | `NPF_LASTQUAD_INIT_EPS` now represents the quadratic coefficient scale; the trainable quadratic factor is initialized at `sqrt(init_eps)`. | The old factor initialization made the final quadratic contribution scale like `init_eps^2`; with `1e-4`, the actual coefficient was `1e-8` and `q_out.delta_raw` barely moved. |
| BB wrong-curvature fallback | When the ascent BB secant has the wrong curvature sign, the step proposal falls back to `BB_ALPHA_MIN` instead of reusing a stale `alpha_prev`. | Reusing a stale large step can keep forcing unstable adversary updates after the stochastic local objective becomes noisy or locally convex. |
| BB adversary gradient clipping | Shared parametric BB adversaries use `PARAMETRIC_BB_MAX_GRAD_NORM=1.0` by default. | This restores the legacy ICNN behavior (`clip_grad_norm_(icnn.parameters(), max_norm=1.0)`) before the BB/Armijo update. |
| Clean-correct mask under DDP | NPF and NN-DRO now optimize the globally masked objective by reducing the clean-correct count across ranks. | The old DDP path averaged each rank's masked mean equally, biasing the adversary gradient when ranks had different numbers of clean-correct samples. |
| Epoch diagnostics | `theta_l2_delta` and `omega_l2_delta` are printed each epoch and written to `epoch_log.csv`. | Tracks `||theta_t - theta_{t-1}||_2` for the classifier and `||omega_t - omega_{t-1}||_2` for the NPF adversary. |
| Interrupted jobs | Epoch CSV rows are appended at epoch end, not only after `fit()` returns. | Completed epochs remain analyzable even if the cluster job is interrupted later. |

The ResNet loader unwraps `R2.pth` using the shared checkpoint priority in `utils.unwrap_state_dict`. For the current `R2.pth`, whose top-level keys are `last`, `best`, `swa_last`, and `swa_best`, the loaded classifier branch is `last`.

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
| Legacy semantics | `PARAMETRIC_BB_MAX_GRAD_NORM` | `1.0` | Clip shared parametric BB adversary gradients before BB/Armijo. |
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
    PARAMETRIC_BB_MAX_GRAD_NORM=1.0 \
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

## Corrected K10 low-LR fixed-lambda command

Use this command for the low-learning-rate K10 diagnostic after the NPF/BB fixes above. The run names encode `lr0p01`, `K10`, the legacy BB fixes, and gradient clipping.

```bash
python csub.py -n lastquad-lam30-logsum-bb-k10-lr0p01-fix -g 2 -t 1d --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/LAT && \
    source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    RUN_NAME=npf_lq_lam30_logsum_lr0p01_K10_bb_legacyfix_clip1_pgdce_seed1 \
    LR_THETA=0.01 \
    PENALTY_LAMBDA=30 \
    TRANSPORT_COST=normalized_mse \
    EPOCHS_ICNN_PRETRAIN=0 \
    LAMBDA_SCHEDULE='' \
    LAMBDA_STAGE_EPOCHS=0 \
    ATTACK_CLEAN_CORRECT_ONLY=1 \
    RESET_PARAMETRIC_BB_EACH_BATCH=1 \
    PARAMETRIC_BB_MAX_GRAD_NORM=1.0 \
    FREEZE_BATCHNORM=0 \
    FREEZE_BATCHNORM_AFFINE=0 \
    BATCHNORM_ONLINE_REFRESH=0 \
    RECALIBRATE_BATCHNORM=0 \
    USE_MARGIN_LOSS=1 \
    COMMON_BATCH=512 \
    NPF_LASTQUAD_HIDDEN='1024 512 512 256 128 64' \
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
    OMEGA_STEPS=10 \
    FROZEN_ADVERSARY_EPOCHS=0 \
    FROZEN_ADVERSARY_MAP_STEPS=1 \
    OUTPUT_FOLDER_NAME=BB_lam30_legacyfix_logsum_lr0p01_K10_clip1_pgdce_seed1 \
    bash run_runtime_sweep_ddp.sh 0 1 2 10 50"
```

### Exact legacy-script comparison knobs

For a closer comparison to the `26036bb` shell script, change only these fields after the above baseline is reproducible:

```bash
EPOCHS_ICNN_PRETRAIN=2
OUTPUT_FOLDER_NAME=BB_lam30_legacyfix_logsum_lr0p1_K7_warm2_pgdce_seed1
RUN_NAME=npf_lq_lam30_logsum_lr0p1_K7_bb_stablepolicy_warm2_pgdce_seed1
bash run_runtime_sweep_ddp.sh 0 1 2 7 30
```

## Lambda-schedule ablation: descending lambda, K=20

This ablation uses the stable-policy run settings, but replaces the fixed `lambda=30` objective with a descending schedule. The training process is continuous: `pretrained_input_icnn` updates the active `lambda_param` at epoch boundaries while keeping the ResNet-18 classifier weights, the NPF adversary weights, and the NPF optimizer/adversary state alive.

The effective inner budget is `K=20`, because both `OMEGA_STEPS=20` and the final launcher call `bash run_runtime_sweep_ddp.sh 0 1 2 20 50` set the NPF adversary updates per batch to 20.

| Group | Setting | Value | Notes |
| --- | --- | --- | --- |
| Lambda schedule | `LAMBDA_SCHEDULE` | `30 20 15 10 5` | Active lambda values in order. |
| Lambda schedule | `LAMBDA_STAGE_EPOCHS` | `10` | Each lambda is used for 10 adversarial epochs. |
| Lambda schedule | epoch 1-10 | `30` | Starts from the stable fixed-lambda value. |
| Lambda schedule | epoch 11-20 | `20` | Same adversary and classifier continue training. |
| Lambda schedule | epoch 21-30 | `15` | No restart between stages. |
| Lambda schedule | epoch 31-40 | `10` | NPF weights are retained and adapted. |
| Lambda schedule | epoch 41-50 | `5` | Final low-penalty stage. |
| Launcher | `K` / `OMEGA_STEPS` | `20` | Twenty NPF adversary updates per batch. |
| Launcher | adversarial epochs | `50` | Must equal `len(schedule) * stage_epochs`. |
| Classifier | `LR_THETA` | `0.1` | Stable-policy classifier learning rate. |
| DRO | `PENALTY_LAMBDA` | `30` | Initial/print value; Python uses `LAMBDA_SCHEDULE` once provided. |
| DRO | `TRANSPORT_COST` | `normalized_mse` | Legacy normalized-coordinate mean squared transport cost. |
| Legacy semantics | `ATTACK_CLEAN_CORRECT_ONLY` | `1` | Attack only clean-correct samples. |
| Legacy semantics | `RESET_PARAMETRIC_BB_EACH_BATCH` | `1` | Reset BB secant history each batch. |
| Legacy semantics | `PARAMETRIC_BB_MAX_GRAD_NORM` | `1.0` | Clip shared parametric BB adversary gradients before BB/Armijo. |
| BatchNorm | `FREEZE_BATCHNORM` | `0` | Stable-policy BatchNorm behavior. |
| BatchNorm | `FREEZE_BATCHNORM_AFFINE` | `0` | BN affine parameters remain trainable. |
| BatchNorm | `BATCHNORM_ONLINE_REFRESH` | `0` | No clean BN refresh pass. |
| BatchNorm | `RECALIBRATE_BATCHNORM` | `0` | No post-epoch BN recalibration. |
| NPF architecture | `NPF_LASTQUAD_HIDDEN` | `1024 512 512 256` | Same LastQuad width as stable-policy run. |
| NPF architecture | `NPF_LASTQUAD_SOFTPLUS_BETA` | `20.0` | Same softplus beta as stable-policy run. |
| NPF optimizer | `NPF_INNER_OPTIMIZER` | `bb_armijo` | Custom BB+Armijo gradient ascent on NPF weights. |
| NPF optimizer | `BB_ALPHA0`, `BB_ALPHA_MIN`, `BB_ALPHA_MAX` | `5e-4`, `1e-6`, `1.0` | BB step-size controls. |
| NPF optimizer | `BB_LS_C`, `BB_LS_SHRINK`, `BB_LS_MAX_STEPS` | `0.1`, `0.5`, `10` | Armijo line-search controls. |
| PGD eval | `INPUT_PGD_LOSS` | `ce` | Cross-entropy PGD evaluation. |
| PGD eval | `EVAL_PGD_SAMPLES` | `2000` | Test samples used per epoch. |

Use this cleanly named command for reproducible reruns:

```bash
python csub.py -n lastquad-lamsched-30-20-15-10-5-k20-lr0p1 -g 2 -t 1d --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/LAT && \
    source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    RUN_NAME=npf_lq_lamsched_30_20_15_10_5_stage10_logsum_lr0p1_K20_bb_stablepolicy_pgdce_seed1 \
    LR_THETA=0.1 \
    PENALTY_LAMBDA=30 \
    TRANSPORT_COST=normalized_mse \
    EPOCHS_ICNN_PRETRAIN=0 \
    LAMBDA_SCHEDULE='30 20 15 10 5' \
    LAMBDA_STAGE_EPOCHS=10 \
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
    PARAMETRIC_BB_MAX_GRAD_NORM=1.0 \
    EVAL_PGD_SAMPLES=2000 \
    INPUT_PGD_LOSS=ce \
    INP_STEPS=20 \
    INP_RESTARTS=5 \
    OMEGA_STEPS=20 \
    FROZEN_ADVERSARY_EPOCHS=0 \
    FROZEN_ADVERSARY_MAP_STEPS=1 \
    OUTPUT_FOLDER_NAME=BB_lamsched_30_20_15_10_5_stage10_logsum_lr0p1_K20_pgdce_seed1 \
    bash run_runtime_sweep_ddp.sh 0 1 2 20 50"
```

Artifact note: the originally submitted command used names containing `5_10_15_20_30` and `K7`, but the actual executed hyperparameters were `LAMBDA_SCHEDULE='30 20 15 10 5'`, `OMEGA_STEPS=20`, and launcher `K=20`.

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
  --parametric-bb-max-grad-norm 1.0 \
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
python -m pytest tests
```
