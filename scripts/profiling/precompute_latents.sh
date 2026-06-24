#!/bin/bash
# precompute_latents.sh — 阶段1:把数据预编码成 latent 缓存(VAE/文本编码只做一次)
# 之后 run_wan_train_cached_nvtx.sh 训练时直接读缓存,跳过 VAE。
# 用法: bash precompute_latents.sh   (一次性,不长)

set -e
cd /root/DiffSynth-Studio

MODELS='["/root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors","/root/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","/root/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]'

python examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv \
  --height 480 --width 832 --dataset_repeat 1 \
  --model_paths "$MODELS" \
  --output_path ./data/cache_wan_lora \
  --task "sft:data_process"

echo ""
echo "=== 预编码完成,latent 缓存在 ./data/cache_wan_lora ==="
ls -R ./data/cache_wan_lora 2>/dev/null | head
