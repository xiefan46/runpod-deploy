#!/bin/bash
# install_nsight.sh — 装 Nsight Systems (nsys) + Nsight Compute (ncu)
#
# devel 镜像通常已带(CUDA toolkit 里),先检测;没有再装。
# 用法: bash install_nsight.sh

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }

if command -v nsys &>/dev/null; then log "nsys 已装: $(nsys --version | head -1)"; else
    log "安装 Nsight Systems..."
    apt-get update -qq && apt-get install -y nsight-systems-cli 2>/dev/null || \
        log "apt 装 nsys 失败,可从 NVIDIA 官网下 .run 包,或用 CUDA toolkit 自带的 nsys"
fi

if command -v ncu &>/dev/null; then log "ncu 已装: $(ncu --version | head -1)"; else
    log "安装 Nsight Compute..."
    apt-get install -y nsight-compute 2>/dev/null || \
        log "apt 装 ncu 失败,CUDA toolkit 自带 ncu(在 /usr/local/cuda/bin/ 或 /opt/nvidia/nsight-compute/)"
fi

echo ""
log "检测结果:"
command -v nsys &>/dev/null && echo "  nsys: $(which nsys)" || echo "  nsys: 未找到(profiling Step1-3/6 纯 torch 不需要它,可先跳过)"
command -v ncu  &>/dev/null && echo "  ncu : $(which ncu)"  || echo "  ncu : 未找到(同上,可后续再装)"
echo ""
echo -e "${YELLOW}nsys 用法: nsys profile -o myrun python xxx.py${RESET}"
echo -e "${YELLOW}ncu  用法: ncu --set full --launch-count 8 -o myncu python xxx.py${RESET}"
