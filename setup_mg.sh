#!/bin/bash
# setup_mg.sh — clone Matrix-Game(+ Wan)用于推理对照
# 对应 hands-on M2(推理)
# 用法: bash setup_mg.sh
# 注意: MG-1/2/3 各子项目依赖不同,按对应子项目 README 装 requirements。

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }

MG_DIR="${MG_DIR:-/root/Matrix-Game}"
[ -d "$MG_DIR" ] || { log "clone Matrix-Game..."; git clone --depth 1 https://github.com/SkyworkAI/Matrix-Game.git "$MG_DIR"; }

WAN_DIR="${WAN_DIR:-/root/Wan2.2}"
[ -d "$WAN_DIR" ] || { log "clone Wan2.2..."; git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "$WAN_DIR"; }

echo ""
log "已 clone:"
echo "  Matrix-Game: $MG_DIR  (MG-1 / MG-2 / MG-3 三个子项目)"
echo "  Wan2.2:      $WAN_DIR"
echo ""
echo -e "${YELLOW}装依赖(按子项目):${RESET}"
echo "  cd $MG_DIR/Matrix-Game-2 && pip install -r requirements.txt   # MG-2 推理"
echo "  cd $MG_DIR/Matrix-Game-3 && pip install -r requirements.txt   # MG-3 推理"
echo -e "${YELLOW}权重: 见各子项目 README 的 HF 链接,或 bash download_models.sh <repo>${RESET}"
echo -e "${YELLOW}M2 练习: 跑 MG/FastWan 推理 → 用 profiling 工具看瓶颈(对照 MG-3 Table 1)${RESET}"
