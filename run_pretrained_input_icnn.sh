#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --pretrained-path /mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints/R2.pth
  --visualize-transport
  --transport-viz-method tsne
  --icnn-activation softplus
  --icnn-hidden 1024 512 512 256
  --estimate-transport-jacobian
  --icnn-init principled
  --eval-input-pgd
  --eval-input-pgd-samples 1000
  --jacobian-aware
  --track-transport-deltas
  --icnn-ascent-steps 7
  --icnn-strong-convexity 1.0
  --icnn-step-rule bb-armijo
  --penalty-lambda 30
  --epochs-icnn-pretrain 2
  --epochs-adv-finetune 0
  --epochs-adv 30
  --epochs-clean 0
  --lr-omega 0.0005
  --lr-theta 0.1
  --icnn-optimizer sgd
  --adv-image-every-epoch
  --visualize-adversarial-images
  --adv-image-require-fooling
  --adv-image-samples 30
  --batch-size 512
  --use-margin-loss
  --inp-p "2"
  --inp-eps 0.5
)

CALIBRATION_ARGS=()  # Calibration disabled for both ICNN and WRM.

for METHOD in icnn; do
  SAVE_PATH="R2_INPUT_${METHOD}_lambda_30_epochs_adv_30_l2_PGD.pth"
  python pretrained_INPUT_icnn.py \
    --adv-method "${METHOD}" \
    --save "${SAVE_PATH}" \
    "${COMMON_ARGS[@]}" \
    "${CALIBRATION_ARGS[@]}"
done
