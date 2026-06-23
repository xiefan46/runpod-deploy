# runpod-deploy

RunPod 上一键 setup **世界模型 infra 实操环境**(profiling + 各框架),用于训练/微调/推理的 hands-on 练习。

> 参考 verl-deploy 的模式,但内容面向视频扩散/世界模型(FastVideo / DiffSynth-Studio / Matrix-Game / Wan)+ profiling 工具链。
> **所有脚本在 `scripts/` 下。**

## RunPod 基础镜像

创建 Pod 时 Docker Image 填:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

(含 PyTorch 2.8 + CUDA 12.8 + cuDNN;`devel` 变体带 nvcc,能编 flash-attn 等。B 系列卡用更新的 CUDA 镜像。)

> 镜像自带 torch+CUDA,**直接用 base python,不另建 conda env**(最省事)。各脚本 pip 安装在 base 之上。

## 快速开始

```bash
# 1. clone 本仓库
git clone https://github.com/xiefan46/runpod-deploy.git /root/runpod-deploy
cd /root/runpod-deploy

# 2. 基础环境(系统工具 + 公共依赖 + 验证 torch/CUDA)
bash scripts/setup_env.sh

# 3. 先跑 profiling 入门(纯 torch,零额外安装)—— Module 0
cd scripts/profiling
python run_basic.py        # 吞吐 + 峰值显存
python run_torchprof.py    # op 热点(终端表 + trace.json)
python run_mfu.py          # MFU%
python run_memsnap.py      # 显存快照 → 上传 pytorch.org/memory_viz
```

## 各模块 setup(均在 `scripts/`)

| 想做 | 脚本 | 说明 |
|------|------|------|
| 基础环境 | `scripts/setup_env.sh` | 系统工具 + huggingface_hub/wandb 等 + 验证 GPU |
| profiling 工具(nsys/ncu) | `scripts/install_nsight.sh` | 装 Nsight Systems/Compute(系统级 + kernel 级) |
| 推理框架 FastVideo | `scripts/setup_fastvideo.sh` | clone + 安装(STA/DMD/self-forcing,M2/M4) |
| 微调框架 DiffSynth-Studio | `scripts/setup_diffsynth.sh` | clone + 安装(Wan LoRA/全量微调,M1) |
| Matrix-Game / Wan | `scripts/setup_mg.sh` | clone(推理对照,M2) |
| 下载模型权重 | `scripts/download_models.sh [repo]` | 默认 Wan2.1-T2V-1.3B |

## 典型流程(对应 hands-on 计划)

```bash
bash scripts/setup_env.sh
# M0 profiling:
cd scripts/profiling && python run_basic.py && python run_torchprof.py && python run_mfu.py
# M1 微调(从 scripts/profiling 用 ../ 即回到 scripts/):
bash ../setup_diffsynth.sh && bash ../download_models.sh
# M2 推理:
bash ../setup_fastvideo.sh   # 跑 FastWan 推理 + 用同样的 profiling 工具
```

## 监控

```bash
# GPU 占用
watch -n 1 nvidia-smi
# wandb(一次性登录)
wandb login
# 本地看 trace:把 trace.json / snap.pickle scp 回本地
#   trace.json  → ui.perfetto.dev 或 chrome://tracing
#   snap.pickle → https://pytorch.org/memory_viz
# SSH 端口转发(如需看 dashboard)
ssh -L 8265:localhost:8265 root@<host> -p <port> -i ~/.ssh/id_ed25519
```

## 说明

- **不另建 conda env**:直接用镜像自带 torch(profiling 够用)。框架若依赖冲突,可按需 `python -m venv` 隔离。
- **`scripts/profiling/`**:Module 0 的可跑脚本(mini 模型 + 各工具),换靶子(真实模型)时复用同样的工具。
- **缓存策略(后续优化)**:若框架安装变慢,可仿 verl-deploy 把 env 打包推 HF Hub dataset 加速(目前先不做)。
- 详细 profiling 教程见 skywork-job 仓库 `research/module0-profiling-guide.md`。
