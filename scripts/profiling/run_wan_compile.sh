#!/bin/bash
# run_wan_compile.sh — 读缓存训练 + torch.compile DiT,对比 it/s(无损加速杠杆)
#
# 前提:先跑过 precompute_latents.sh(缓存在 ./data/cache_wan_lora)
#       且 DiffSynth fork 已更新到带 --use_torch_compile 的版本(git pull)
# 对比基准:不带 compile 的读缓存训练 ≈ 7.9 s/it
# ⚠️ compile 首步很慢(在编译,几十秒~几分钟),看【稳态】it/s(让它跑 20+ 步)

set -e
cd /root/DiffSynth-Studio

python examples/wanvideo/model_training/train.py \
  --dataset_base_path ./data/cache_wan_lora \
  --dataset_repeat 200 --num_epochs 1 \
  --model_paths '["/root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"]' \
  --learning_rate 1e-4 \
  --output_path ./models/train/Wan2.1-T2V-1.3B_lora_compile \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --task "sft:train" \
  --use_torch_compile

echo ""
echo "=== 对比:不带 compile ~7.9 s/it;compile 首步慢(编译),看稳态 it/s。跑够 20 步 Ctrl-C ==="
