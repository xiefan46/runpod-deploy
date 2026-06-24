#!/bin/bash
# run_wan_cache_offload_nvtx.sh — cache + 逐层 offload(极致省显存)
#   - 读缓存 latent(VAE/T5 已剪)
#   - 唯一剩下的 DiT 也 --enable_model_cpu_offload 逐层 offload
# 前提:跑过 precompute_latents.sh
# 看:nvidia-smi(另开终端,显存应是所有配置里最低)+ it/s + overlap

set -e
cd /root/DiffSynth-Studio

nsys profile --trace=cuda,nvtx,osrt --delay 50 --duration 100 \
  -o wan_cache_offload_nvtx --force-overwrite true \
  python examples/wanvideo/model_training/train.py \
  --dataset_base_path ./data/cache_wan_lora \
  --dataset_repeat 100 --num_epochs 1 \
  --model_paths '["/root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"]' \
  --learning_rate 1e-4 \
  --output_path ./models/train/Wan2.1-T2V-1.3B_lora_cache_offload \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --task "sft:train" \
  --enable_model_cpu_offload

echo ""
echo "=== 看 overlap: ==="
echo "nsys stats wan_cache_offload_nvtx.nsys-rep"
echo "python /root/runpod-deploy/scripts/profiling/overlap_analysis.py wan_cache_offload_nvtx.sqlite"
