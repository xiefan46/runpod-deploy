#!/bin/bash
# setup_diffsynth.sh — clone + 安装 DiffSynth-Studio(Wan LoRA/全量微调,最易上手)
# 对应 hands-on M1(微调)
# 用法: bash setup_diffsynth.sh

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }

REPO_DIR="${DIFFSYNTH_DIR:-/root/DiffSynth-Studio}"
[ -d "$REPO_DIR" ] || { log "clone DiffSynth-Studio..."; git clone --depth 1 https://github.com/modelscope/DiffSynth-Studio.git "$REPO_DIR"; }
cd "$REPO_DIR"

log "安装 DiffSynth-Studio(editable)..."
pip install -e . || { log "pip install -e . 失败,试 requirements"; pip install -r requirements.txt 2>/dev/null || true; }

log "验证 import..."
python -c "import diffsynth; print('DiffSynth OK')" 2>/dev/null || log "import 失败,按 repo README 排依赖"

echo ""
echo -e "${YELLOW}Wan 微调示例: $REPO_DIR/examples/wanvideo/model_training/${RESET}"
echo -e "${YELLOW}先下模型: bash download_models.sh${RESET}"
echo -e "${YELLOW}M1 练习: LoRA 微调 Wan-1.3B 跑通 → 量显存/吞吐 → 开 FP8/offload 看变化${RESET}"
