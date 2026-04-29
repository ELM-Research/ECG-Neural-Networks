CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data batch10 batch8 batch9 \
--data_representation "signal" \
--objective "st_mem" \
--neural_network "st_mem" \
--task "pretrain" \
--batch_size 64 \
--distributed \
--ref_global_bs 512 \
--epochs 300 \
--torch_compile \
--lr 3e-4 \
--lr_schedule cosine \
--weight_decay 1e-2 \
--beta1 0.9 \
--beta2 0.95 \
--augment \
--warmup 10000 \
--optimizer adamw \
--grad_clip 1.0 \
--num_workers 16 \
--wandb