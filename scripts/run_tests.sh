echo "==> Running CPU unit tests"
pytest -q tests/dir_file.py -s

echo "==> Running single-GPU + FlashAttention tests"
pytest -q tests/single_gpu.py -k "test_cuda_available or test_basic_tensor_ops" -s

echo "==> To run DDP test (example with 2 GPUs):"
echo "torchrun --standalone --nproc_per_node=2 -m pytest -q -s tests/multi_gpu.py::test_ddp_allreduce"