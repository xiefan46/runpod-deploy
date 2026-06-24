#!/bin/bash
# run_wan_nvtx.sh — nsys profile Wan2.1-T2V-1.3B LoRA 训练(带 NVTX 阶段标签),抓几步
#
# 前提:
#   - DiffSynth fork 的 nvtx-profiling 分支在 /root/DiffSynth-Studio
#   - 模型在 /root/models/Wan2.1-T2V-1.3B
#   - 示例数据集已下到 /root/DiffSynth-Studio/data/diffsynth_example_dataset
#
# 用法: bash run_wan_nvtx.sh
# 看结果: nsys stats wan_train_nvtx.nsys-rep   (顶部应有 NVTX Range Summary)

set -e
cd /root/DiffSynth-Studio

MODELS='["/root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors","/root/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","/root/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]'

nsys profile --trace=cuda,nvtx,osrt --delay 75 --duration 90 \
  -o wan_train_nvtx --force-overwrite true \
  python examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv \
  --height 480 --width 832 --dataset_repeat 100 \
  --model_paths "$MODELS" \
  --learning_rate 1e-4 --num_epochs 1 \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32

echo ""
echo "=== nsys 采集完成。看阶段表: ==="
echo "nsys stats wan_train_nvtx.nsys-rep"
