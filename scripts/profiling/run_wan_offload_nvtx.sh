#!/bin/bash
# run_wan_offload_nvtx.sh — full LoRA + 逐层 offload(--enable_model_cpu_offload),nsys 抓几步
# 目的:看 compute(kernel)和 offload 的 transfer(memcpy)能不能互相掩盖(overlap)。
# 看结果:nsys stats + overlap_analysis.py(量化 overlap%,不用 GUI)

set -e
cd /root/DiffSynth-Studio

MODELS='["/root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors","/root/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","/root/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]'

# offload 模式 step 慢、加载方式不同,窗口给宽一点;如没抓到步数就调 --delay/--duration
nsys profile --trace=cuda,nvtx,osrt --delay 60 --duration 150 \
  -o wan_offload_nvtx --force-overwrite true \
  python examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv \
  --height 480 --width 832 --dataset_repeat 100 \
  --model_paths "$MODELS" \
  --learning_rate 1e-4 --num_epochs 1 \
  --output_path ./models/train/Wan2.1-T2V-1.3B_lora_offload \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --enable_model_cpu_offload

echo ""
echo "=== 看 overlap: ==="
echo "nsys stats wan_offload_nvtx.nsys-rep    # 生成 .sqlite + 看 memcpy/kernel/NVTX"
echo "python /root/runpod-deploy/scripts/profiling/overlap_analysis.py wan_offload_nvtx.sqlite"
