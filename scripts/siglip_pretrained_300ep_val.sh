#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation signal \
--objective siglip2 \
--neural_network siglip2-base-patch16-naflex \
--task pretrain \
--seed 0 \
--batch_size 256 \
--distributed \
--ref_global_bs 1024 \
--epochs 300 \
--val_split 20000 \
--save_epoch 0 \
--lr 8e-5 \
--lr_schedule cosine \
--warmup_ratio 0.00833 \
--weight_decay 1e-8 \
--beta1 0.9 \
--beta2 0.999 \
--grad_clip 3.0 \
--text_feature_extractor google/siglip2-base-patch16-naflex \
--condition_text_max_len 64 \
--num_workers 16 \
--wandb
