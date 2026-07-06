#!/usr/bin/env bash
set -euo pipefail

python pretrained_INPUT_icnn.py \
  --pretrained-path /mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints/R2.pth \
  --visualize-transport \
  --icnn-activation softplus \
  --icnn-hidden 1024 512 512 256 128 64 \
  --icnn-init principled \
  --eval-input-pgd \
  --eval-input-pgd-samples 1000 \
  --track-transport-deltas \
  --icnn-ascent-steps 10 \
  --icnn-strong-convexity 1.0 \
  --icnn-step-rule bb-armijo \
  --penalty-lambda 30 \
  --epochs-icnn-pretrain 2 \
  --epochs-adv-finetune 0 \
  --epochs-adv 30 \
  --epochs-clean 0 \
  --lr-omega 0.0005 \
  --lr-theta 0.1 \
  --icnn-optimizer sgd \
  --adv-image-every-epoch \
  --visualize-adversarial-images \
  --adv-image-require-fooling \
  --adv-image-samples 30 \
  --batch-size 512 \
  --use-margin-loss \
  --inp-p "2" \
  --inp-eps 0.5 \
  --save R2_lambda_30_icnnpretrain_2_minimax30_Final.pth
  