#!/usr/bin/env bash
set -euo pipefail

python pretrained_INPUT_icnn.py \
  --pretrained-path /mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints/R2.pth \
  --visualize-transport \
  --transport-viz-method tsne \
  --icnn-activation softplus \
  --icnn-hidden 1024 512 512 256 \
  --estimate-transport-jacobian \
  --icnn-init principled \
  --eval-input-pgd \
  --eval-input-pgd-samples 1000 \
  --jacobian-aware \
  --track-transport-deltas \
  --icnn-ascent-steps 10 \
  --icnn-strong-convexity 1.0 \
  --icnn-step-rule bb-armijo \
  --penalty-lambda 5 \
  --epochs-icnn-pretrain 0 \
  --epochs-adv-finetune 0 \
  --epochs-adv 50 \
  --epochs-clean 0 \
  --use-margin-loss \
  --adv-image-every-epoch \
  --visualize-adversarial-images
