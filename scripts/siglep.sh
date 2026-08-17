CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data ELM-Research/pretrain-encoder \
--data_representation "signal" \
--objective "siglip2" \
--neural_network "siglip2-base-patch16-naflex" \
--patch_size 25 \
--task "pretrain" \
--batch_size 1024 \
--distributed \
--ref_global_bs 8192 \
--epochs 300 \
--torch_compile \
--lr 3e-4 \
--lr_schedule cosine \
--weight_decay 1e-2 \
--beta1 0.9 \
--beta2 0.95 \
--augment \
--optimizer adamw \
--grad_clip 1.0 \
--condition_text_max_len 64 \
--num_workers 16 \
--text_feature_extractor google/siglip2-base-patch16-naflex \
--save_epoch 50 \
--wandb
