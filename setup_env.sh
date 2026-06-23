#!/bin/bash
# setup_env.sh — RunPod 基础环境(世界模型 infra 实操)
#
# 直接用镜像自带 torch/CUDA(runpod/pytorch:...),不另建 conda env。
# 装系统工具 + 公共 python 依赖 + 验证 GPU。
#
# 用法: bash setup_env.sh

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()  { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

export DEBIAN_FRONTEND=noninteractive

# ─── 系统工具 ───
NEED=""
for t in tmux git wget zstd htop; do command -v $t &>/dev/null || NEED="$NEED $t"; done
if [ -n "$NEED" ]; then
    log "安装系统工具:$NEED"
    apt-get update -qq && apt-get install -y $NEED
fi

# ─── 公共 python 依赖(装进 base python)───
log "安装公共 python 依赖..."
pip install -qU pip
pip install -q "huggingface_hub[cli]" hf_transfer wandb calflops 2>/dev/null || \
    pip install -q "huggingface_hub[cli]" hf_transfer wandb
export HF_HUB_ENABLE_HF_TRANSFER=1

# ─── 验证 GPU ───
log "验证环境..."
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
assert torch.cuda.is_available(), "CUDA not available!"
print(f"CUDA: {torch.version.cuda}")
n = torch.cuda.device_count()
print(f"GPUs: {n} x {torch.cuda.get_device_name(0)}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB  SM {p.major}.{p.minor}")
print("=== torch profiler 自带,可直接用 ===")
PY

echo -e "\n${BOLD}${GREEN}基础环境就绪${RESET}"
echo -e "${YELLOW}下一步:cd profiling && python run_basic.py${RESET}"
echo -e "${YELLOW}装 nsys/ncu: bash install_nsight.sh${RESET}"
echo -e "${YELLOW}装框架: bash setup_fastvideo.sh / setup_diffsynth.sh / setup_mg.sh${RESET}\n"
