#!/bin/bash
# download_models.sh — 从 HF 下载模型权重
# 用法:
#   bash download_models.sh                          # 默认 Wan2.1-T2V-1.3B(小,适合 M1 微调/M0 上手)
#   bash download_models.sh Wan-AI/Wan2.2-TI2V-5B    # 指定 repo
#   bash download_models.sh org/repo /root/models/x  # 指定本地路径

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }

REPO="${1:-Wan-AI/Wan2.1-T2V-1.3B}"
DEST="${2:-/root/models/$(basename "$REPO")}"

command -v hf &>/dev/null || pip install -q "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 私有/受限 repo 需登录(公开的无需)
hf auth whoami &>/dev/null || log "(如是私有 repo,先 hf auth login;公开 repo 可忽略)"

log "下载 $REPO → $DEST"
mkdir -p "$DEST"
SECONDS=0
hf download "$REPO" --local-dir "$DEST"
log "完成 (${SECONDS}s): $(du -sh "$DEST" | cut -f1)"
echo -e "${YELLOW}路径: $DEST${RESET}"
