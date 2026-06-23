#!/bin/bash
# install_nsight.sh — 装 Nsight Systems (nsys) + Nsight Compute (ncu)
#
# devel 镜像通常已带(CUDA toolkit 里),先检测;没有再装。
# 用法: bash install_nsight.sh

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }

# nsys: apt 的 nsight-systems meta 在部分镜像会拉到坏的旧版(2022,缺 importer,无法生成 .nsys-rep)。
# 策略:PATH 里若没有或不是现代版(2023+),就用 /opt/nvidia 下的完整版(nsight-systems 或 ncu 自带)symlink 过去。
if command -v nsys &>/dev/null && nsys --version 2>/dev/null | grep -qE '20(2[3-9]|[3-9][0-9])'; then
    log "nsys 已装(现代版): $(nsys --version | tail -1)"
else
    log "PATH 无可用 nsys,找盘上完整版..."
    FOUND="$(find /opt/nvidia -name nsys -type f 2>/dev/null | sort -V | tail -1)"
    if [ -n "$FOUND" ]; then
        ln -sf "$FOUND" /usr/local/bin/nsys
        log "已 symlink: $FOUND → /usr/local/bin/nsys ($(nsys --version | tail -1))"
    else
        apt-get update -qq && apt-get install -y nsight-systems-2025.5.2 || \
            log "nsys 装失败:apt-cache search nsight 看可用版本,或官网下 .run"
    fi
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
