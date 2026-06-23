#!/bin/bash
# setup_fastvideo.sh — clone + 安装 FastVideo(推理加速 + DMD/self-forcing 蒸馏 + STA)
# 对应 hands-on M2(推理)/ M4(蒸馏)
# 用法: bash setup_fastvideo.sh

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }

REPO_DIR="${FASTVIDEO_DIR:-/root/FastVideo}"
[ -d "$REPO_DIR" ] || { log "clone FastVideo..."; git clone --depth 1 https://github.com/hao-ai-lab/FastVideo.git "$REPO_DIR"; }
cd "$REPO_DIR"

log "安装 FastVideo(editable)..."
# 官方推荐 uv;没有 uv 退回 pip。若依赖与镜像 torch 冲突,按 repo README 调整。
if command -v uv &>/dev/null; then
    uv pip install -e . || pip install -e .
else
    pip install -e . || { log "pip install -e . 失败,试 requirements"; pip install -r requirements.txt 2>/dev/null || true; }
fi

log "验证 import..."
python -c "import fastvideo; print('FastVideo OK')" 2>/dev/null || log "import 失败,按 FastVideo README 排依赖"

echo ""
echo -e "${YELLOW}STA kernel(可选,M2 稀疏 attention):cd $REPO_DIR/fastvideo-kernel && ./build.sh${RESET}"
echo -e "${YELLOW}推理 demo: 见 $REPO_DIR/examples/inference/${RESET}"
echo -e "${YELLOW}蒸馏代码(读): $REPO_DIR/fastvideo/train/methods/distribution_matching/{dmd2.py,self_forcing.py}${RESET}"
