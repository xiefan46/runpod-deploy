# CLAUDE.md — runpod-deploy 说明(给 agent / 未来的自己)

## 这是什么
RunPod 上 setup **世界模型 infra 实操环境**的脚本集——profiling + 框架(FastVideo / DiffSynth-Studio / Matrix-Game / Wan)。配合 skywork-job 仓库的 `research/hands-on-plan.md`(5-Module 实操课程)和 `research/module0-profiling-guide.md` 用。

## 约定
- **base image**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`(自带 torch+CUDA,devel 带 nvcc)。
- **不另建 conda env**:直接用 base python pip install。框架若冲突再 `python -m venv` 隔离。
- **框架在 pod 上 runtime clone**,不提交进本仓库(本仓库只放脚本 + profiling 文件,保持轻量)。
- 脚本风格仿 verl-deploy(`set -euo pipefail` + log 函数 + 默认装 /root/)。
- **所有脚本在 `scripts/` 下**(根目录只留 README/CLAUDE/.gitignore)。

## 文件(均在 `scripts/`)
- `setup_env.sh` — 基础(系统工具 + 公共 py 依赖 + 验证 GPU)
- `install_nsight.sh` — nsys/ncu(系统级 + kernel 级 profiling)
- `setup_fastvideo.sh` / `setup_diffsynth.sh` / `setup_mg.sh` — 各框架
- `download_models.sh` — HF 下权重(默认 Wan2.1-T2V-1.3B)
- `profiling/` — Module 0 可跑脚本(mini.py + run_basic/torchprof/memsnap/mfu)

## 流程
`scripts/setup_env.sh` → `scripts/profiling/`(M0,纯 torch,零额外安装)→ 按需 `scripts/setup_diffsynth.sh`(M1)/`scripts/setup_fastvideo.sh`(M2/M4)。

## 后续可加
- 仿 verl-deploy 的 HF Hub env cache(框架装慢时);
- 各框架的 quick-start 食谱(像 verl-deploy README 那样);
- 多卡/多节点 torchrun 食谱;消费级 GPU(量化/offload)的 setup。
