# pretrained_input_icnn — Wasserstein-DRO adversarial training with an NPF-ICNN transport adversary

Input-space adversarial training for CIFAR-10: a pretrained PreActResNet-18
(`ResNet_checkpoints/R2.pth`, 87.3% clean accuracy) is made robust to
pixel-space L2 PGD by solving a Wasserstein-DRO minimax problem whose inner
maximizer is a **learned input-convex transport map** rather than per-sample
PGD. The adversary is an OTT-style NPF ICNN potential ψ_ω; the transport is
its gradient map

```
T_ω(x) = ∇_x ψ_ω(x)
```

trained by BB+Armijo ascent on the penalized objective

```
max_ω  E[ margin(f_θ(T_ω(x)), y) − λ · cost(T_ω(x), x) ],    λ = 30,
cost(x', x) = mean((x'_norm − x_norm)²)      (normalized-MSE, legacy convention)
```

with one classifier SGD step per batch on the transported samples.

The package supersedes the single-file `pretrained_INPUT_icnn_stable_version.py`
(the "legacy golden run", 57% robust accuracy at pixel-L2 ε=0.5 under its own
evaluation convention) and, as of 2026-07-06, **matches and approaches that
benchmark under a strictly harsher evaluator** (see *Results* and *Evaluation
conventions* below).

---

## 1. Results of record (λ=30 normalized-MSE, PGD ε=0.5 / 20 steps / 5 restarts / 1024 test samples, seed 1)

| run | adversary | key deltas | PGD@0.5 (strict) | PGD@0.5 (legacy conv., best ckpt) | clean |
|---|---|---|---|---|---|
| pre-fix era (all runs) | LastQuad, `output_rank=2` | — | **5–16% plateau** | — | 87–91% |
| `ablC` | LastQuad, `output_rank=64` | rank only, old code, LR 0.1, 30 ep | 47.4% final / 48.4% best | 48.34% | 89.3% |
| `flagship` | LastQuad, rank 64 | + all code fixes, LR 0.05, 50 ep | 49.7% final / 49.8% best | 49.80% | 89.2% |
| **`flagshipC`** | **LastQuad, rank 128** | flagship + rank 128 | **52.5% final / 54.2% best** | **54.10%** | **90.1%** |
| legacy golden run | legacy ICNN | single-file implementation | — | 57% (its own convention) | ~87–89% |

The strict evaluator counts `clean-correct AND adv-correct` over all samples
and keeps the best adversarial iterate across **every** PGD step and restart;
the legacy benchmark counted `adv-correct / total`, scoring only each
restart's endpoint. Empirically the two conventions coincide within ~0.1 pt
on these checkpoints, so the remaining gap to 57% is real but small (~3 pts
at rank 128, still shrinking along the rank axis).

---

## 2. Root cause of the historical failure (measured, 2026-07-06 audit)

Every pre-fix run showed the same signature: the adversary devastates the
frozen classifier during warmup (transported accuracy 10–30%), then θ
neutralizes it within one adversarial epoch (transported ≈ clean accuracy)
and PGD robustness plateaus at 5–16%.

**The bottleneck was `output_rank=2`, not BatchNorm and not missing
capacity.** LastQuad carries the same 7.67M trainable per-layer affine input
skips as the legacy ICNN. But each rank-r quadratic direction `a` contributes
`(aᵀx)·a` to T(x) — a *shared* direction whose only sample-dependence is the
scalar coefficient `aᵀx`. Gradient-decomposition measurements on the exact
training configuration (64 clean-correct samples, R2 eval-mode, K=10
BB+Armijo):

| measurement | rank 2 | rank 64 |
|---|---|---|
| ω-gradient energy into final quad + lin + diag (far-field groups) | **>51%** | spread over 64 sample-adaptive directions |
| ω-gradient energy into the deep ladder (`w_zs`) | 1e-4 | 4e-6 |
| `lin_kernel` induced-delta rank-1 energy | 95% (pure translation) | — |
| transported accuracy after the same K=10 ascent | 98% → 64% | 98% → **17%** |
| top-1 SVD energy of learned deltas (PGD reference: ~2%) | 46% | 26% |

At rank 2 the steepest ascent coordinates are literally one or two shared
far-field directions: the ascent grows the objective by dragging every image
along the same low-frequency pattern (measured: pixel-L2 ≈ 3 at 61–67% top-1
SVD energy, cos ≈ 0 to PGD; projecting the delta to ε=0.5 restores ~90%
accuracy). Training θ on such samples teaches it to ignore one global noise
direction — one SGD step suffices — and confers no robustness inside the PGD
ball. BB+Armijo itself was **not** at fault; it faithfully climbed the
objective the parametrization gave it.

A corollary that matters for reading training curves: in *every* successful
run, transported accuracy still hovers near clean accuracy. Robustness does
not come from the adversary "winning" the transported-accuracy race; it comes
from the transports' sample-dependent component overlapping the PGD ball.

## 3. Code fixes applied on top of the rank fix

All verified by a 72-agent audit + adversarial verification and covered by
`tests/test_pretrained_input_icnn_inner_steps.py` (51 tests):

| fix | file | why |
|---|---|---|
| PosDef diag rectifier `relu` → **`exp`** (never-dead), `diag_init` is now effective-valued for every rectifier | `models/npf.py` | the λ-cost gradient pushes raw diag entries negative; `relu` there has exactly zero value **and** gradient forever — a one-way dead zone that killed the elementwise sample-dependent transport path |
| LastQuad output-block diag init 0.1269 → **0.01** | `models/npf.py` | 0.1269 planted a deterministic `T(x) ≈ 1.127x` global-rescale artifact at init (measured 1.54 pixel-L2 = 94% of the init delta, 3× the eval radius): a sample-independent nuisance θ neutralizes in one step |
| PosDef forward via matmul instead of broadcast-and-sum | `models/npf.py` | the broadcast materialized a (B, 3072, width) intermediate — multi-GB per hidden layer in `all_layers` mode, retained for double backward; this is what killed full-NPF runs |
| trainable outer quadratic no longer initialized at exact zeros (constructor **and** identity-init path) | `models/npf.py` | ∂(0.5‖Aᵀx‖²)/∂A ≡ 0 at A=0: an exact saddle, so a zeros-initialized trainable outer quadratic can never train |
| convex-init bias shift sign flipped to **+|bias_mean|** (legacy convention) | `models/npf.py` | the Hoedt–Klambauer derivation gives −0.739; the legacy golden run used +0.739. Sign decides the operating regime: negative leaves 49–68% of deep units active under β=20 softplus, positive runs them 88–100% active — the "hot" regime whose hidden-chain gradients carry sample-dependent attack capacity |
| **ω weight decay 1e-4** restored inside the ascent objective (`--npf-omega-weight-decay`) | `algorithms/npf.py` | legacy applied exactly this to the free (unconstrained) kernels — the affine input skips and PosDef lin/quad — taxing the unbounded growth of the far-field translation/rescale kernels over ~30k persistent-ω ascent steps. Biases, diagonals, the positive `w_zs` ladder, and the frozen outer identity are excluded |
| non-finite batch skip is all-reduced across ranks | `algorithms/base.py` | a rank-local `continue` desynchronizes the DDP gradient all-reduce (hang/corruption) when only one rank overflows |

**Checkpoint-compatibility warning:** ψ_ω checkpoints written before the
`exp`-diag change are not semantically loadable by the new code (raw diag
values now mean `exp(raw)`, previously `relu(raw)`). Classifier checkpoints
are unaffected. Do not `RESUME_CHECKPOINT` a pre-fix adversary.

---

## 4. Reference recipe (= the package defaults)

The flagship recipe below is encoded as the **default at every layer** —
`TrainConfig`, the `argparse` CLI, `run_runtime_sweep_ddp.sh`, and
`run_local_npf_lastquad.sh` are verified field-by-field identical, so a bare
`bash run_local_npf_lastquad.sh` (or even a bare
`python -m pretrained_input_icnn.main`) runs exactly this:

| group | setting | value |
|---|---|---|
| algorithm | `--algorithm` | `npf_lastquad` |
| schedule | adversarial epochs / warmup | **50** / **2** (adversary-only warmup, θ frozen; cosine `T_max` tracks `epochs_adv` automatically) |
| outer (θ) | optimizer | SGD, nesterov 0.9, wd 5e-4, grad-clip 10, **LR 0.05**, global batch **512** |
| DRO objective | λ / cost | **30** on `normalized_mse` (per-sample mean squared difference over the 3072 normalized coordinates; pixel-space equivalent: `pixel_l2_squared` with λ=0.242) |
| adversary loss | `USE_MARGIN_LOSS=1` | logsumexp margin `logsumexp_{j≠y}(logit_j − logit_y)`; CE saturates on clean-correct samples of a confident classifier and stalls the ascent |
| masking | `ATTACK_CLEAN_CORRECT_ONLY=1` | transport trained/applied only on clean-correct samples; clean-incorrect stay clean in the θ update (legacy semantics) |
| inner loop | K (`OMEGA_STEPS`) | **10** BB+Armijo ascent steps per batch, **persistent ω** across batches and epochs (`NPF_RESET_OMEGA_EACH_BATCH=0`), BB secant state reset per batch |
| BB+Armijo | α₀ / α_min / α_max / c / shrink / trials | 5e-4 / 1e-6 / 1.0 / 0.1 / 0.5 / 10; global ω-grad clip **1.0** before the BB proposal (legacy order) |
| ω regularization | `NPF_OMEGA_WEIGHT_DECAY` | **1e-4** on the free kernels (see §3) |
| LastQuad architecture | hidden / activation / rank | `1024 512 512 256 128 64` affine input skips, softplus β=20, **`output_rank=64`** (rank 128 measured better still — see §1), `exp` rectifiers everywhere, output diag init 0.01, strong convexity 1.0 (frozen identity base potential), principled log-normal random init |
| BatchNorm | `FREEZE_BATCHNORM=0`, `ADVERSARY_CLASSIFIER_EVAL=1` | BN participates in the θ update; the adversary attacks the eval-mode (running-stat) classifier — the measured-best policy. `ADVERSARY_CLASSIFIER_EVAL=0` (train-mode attack, exact legacy parity) is a supported ablation arm. Warmup always attacks eval-mode, which **is** legacy parity |
| evaluation | PGD | raw pixel-L2 **ε=0.5**, 20 steps (step 2ε/K), 5 restarts, CE loss, 1024 test samples, eval-mode BN, per-epoch |
| full-NPF arm | `RUN_ONLY_ALGO=npf` | legacy widths, softplus, outer rank 8 / inner rank 2, **identity-adjacent init** (`NPF_IDENTITY_INIT=1`, `NPF_INIT_EPS=1e-2`); the non-identity all-layers init starts 12.9 pixel-L2 from identity at objective −40 and never recovers |

### Single-variable ablation axes (override on top of the defaults)

```bash
LOCAL_NPF_LASTQUAD_OUTPUT_RANK=128   # capacity axis — best measured value so far
LOCAL_K=15                            # inner-iteration axis (also 20)
LOCAL_ADVERSARY_CLASSIFIER_EVAL=0     # train-mode-BN adversary (legacy parity axis)
LOCAL_NPF_LASTQUAD_HIDDEN='128 128 128 128'   # width axis (5× cheaper adversary)
LOCAL_RUN_ONLY_ALGO=npf               # all-layers quadratics architecture family
```

---

## 5. How to run

### Local, single GPU

```bash
RUN_TAG=my_run_name bash run_local_npf_lastquad.sh
```

All knobs are `LOCAL_*` environment variables (see the launcher header).
Launch long runs detached (`setsid nohup ... &`) — interactive-session runs
die with the session. Do **not** edit the launcher or sweep scripts while a
launched run is mid-flight: bash reads scripts incrementally, and an in-place
rewrite corrupts the running interpreter's continuation.

### Cluster, multi-GPU DDP (verified semantics-preserving)

```bash
python csub.py -n lam30-rank128-k20-ddp2 -g 2 -t 1d --train --large-shm --node-type a100-40g \
  --command "cd /mloscratch/homes/aabdolla/LAT && \
    LOCAL_NPROC=2 \
    LOCAL_CUDA_VISIBLE_DEVICES=0,1 \
    LOCAL_K=20 \
    LOCAL_NPF_LASTQUAD_OUTPUT_RANK=128 \
    RUN_TAG=flagshipF_rank128_K20_ddp2_lam30nmse_lr0p05_ep50 \
    bash run_local_npf_lastquad.sh"
```

DDP guarantees (verified in code):

- the train loader shards with `DistributedSampler` at per-rank batch
  `512 / world_size`, so the **global** batch, step count, and LR schedule are
  identical to single-GPU;
- the ω ascent all-reduces both gradients and objective scalars, so every
  rank proposes the same Armijo step and ω stays in lockstep
  (`_shared_adversary_masked_mean` makes the reduced gradient equal the
  gradient of the *globally* masked objective);
- K is per-**global**-batch: GPU count changes wall-clock only, not the
  inner-ascent budget;
- **evaluation runs on rank 0 over the full, unsharded test set** — reported
  robustness numbers mean the same thing at any `NPROC`;
- a non-finite transport on any rank skips the batch on **all** ranks.

Known cosmetic artifact: train-phase CSV columns (`train_loss`, `train_acc`)
log rank-0's shard only. All eval columns are full-set.

### Standalone sweep script

`run_runtime_sweep_ddp.sh` now carries the same defaults as the launcher, but
prefer the launcher: it computes provenance-rich run tags and pins everything
explicitly. Positional contract: `bash run_runtime_sweep_ddp.sh SPLIT SEED
NPROC K ADV_EPOCHS [RUN_TAG]`.

---

## 6. Monitoring and the 10-epoch kill rule

Per-epoch metrics land in `<run>/npf_lastquad/seed_<s>/epoch_log.csv`:

- `input_pgd_acc` (col 44) — the headline robustness number;
- `transport_adv_acc` (col 43) — classifier accuracy **on the NPF transports**.
  Expect this to sit near `clean_acc` after the first adversarial epochs even
  in successful runs (§2); a *decreasing* value late in training means the
  persistent adversary is out-pacing the decaying-LR classifier (healthy);
- `eval_transport_avg_l2` (col 35) — pixel-L2 of the transport deltas;
- `theta_l2_delta` / `omega_l2_delta` — is either player frozen?

**Kill rule:** if `input_pgd_acc` has not risen after 10 adversarial epochs,
terminate the run. (The queue runner in the 2026-07-06 experiments enforces
this automatically: best PGD ≤ first-adv-epoch PGD + 2 pts at adv-epoch ≥ 10
kills the process group.)

## 7. Evaluation conventions

`INPUT_PGD_LOSS` controls only the evaluation attack (`ce` default = the
legacy benchmark convention; `margin` is the stronger local evaluator).
Training-side margin loss is controlled independently by `USE_MARGIN_LOSS`.

When comparing against the legacy 57% figure, remember the two protocol
differences (both make the current numbers *conservative*): the current
evaluator (a) intersects with clean-correctness and (b) keeps the best
adversarial iterate over all steps and restarts, while the legacy evaluator
scored only restart endpoints against `adv-correct/total`.
`scratchpad/legacy_convention_eval.py`-style rescoring of best checkpoints
showed the conventions agree within ~0.1 pt on these models.

## 8. Diagnostics

`ICNN_vs_NPFICNN.ipynb` contains the full
evidence chain: the isolated legacy-vs-NPF adversary comparison, the
"why T(x) fools the model without being adversarial" visual analysis
(delta SVD spectra, batch-mean energy, radial FFT, image grids), the
gradient-decomposition panels (from `debug_outputs/probe_compact.json`), a
live rank-2 vs rank-64 A/B cell, and the training-evidence plot across all
runs of record. `icnn_vs_npficnn_diagnostic.py` provides the underlying
harness.

## 9. Validation

```bash
python -m pytest tests/test_pretrained_input_icnn_inner_steps.py   # 51 tests
bash -n run_local_npf_lastquad.sh run_runtime_sweep_ddp.sh
```

The test suite pins the load-bearing invariants: rank-64 default, `exp` diag
with effective-valued init 0.01, saddle-free trainable outer quadratic,
positive bias shift, BB convex-region proposal, CE eval default, and the
frozen outer identity potential for LastQuad.
