#!/bin/bash
# run_wan_train_cached_nvtx.sh — 阶段2:读预编码 latent 训练(跳过 VAE),nsys+NVTX 抓几步
#
# 前提:先跑过 precompute_latents.sh(缓存在 ./data/cache_wan_lora)
# 对比点:和 wan_train_nvtx.nsys-rep 的 NVTX 阶段表比 ——
#   InputVideoEmbedder(VAE)应该几乎消失,step 明显变快。
#
# 用法: bash run_wan_train_cached_nvtx.sh
# 看结果: nsys stats wan_train_cached_nvtx.nsys-rep

set -e
cd /root/DiffSynth-Studio

# 阶段2 只加载 DiT(VAE/T5 已缓存,不用 load)→ 加载快,所以 delay 小一点
nsys profile --trace=cuda,nvtx,osrt --delay 40 --duration 70 \
  -o wan_train_cached_nvtx --force-overwrite true \
  python examples/wanvideo/model_training/train.py \
  --dataset_base_path ./data/cache_wan_lora \
  --dataset_repeat 100 --num_epochs 1 \
  --model_paths '["/root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"]' \
  --learning_rate 1e-4 \
  --output_path ./models/train/Wan2.1-T2V-1.3B_lora_cached \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --task "sft:train"

echo ""
echo "=== 看阶段表(对比 VAE 是否消失): ==="
echo "nsys stats wan_train_cached_nvtx.nsys-rep"
