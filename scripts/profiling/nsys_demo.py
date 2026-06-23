# nsys_demo.py — 体验 nsys:系统级时间线,看 perfetto 看不到的东西
#   ① NVTX 阶段拆分(数据/传输/前向/反向)→ E2E 各段耗时
#   ② CPU 采样:抓到 numpy 数据准备(perfetto 这里是空白)
#   ③ H2D 传输单独成轨
#
# 跑(pod 上):
#   nsys profile --trace=cuda,nvtx,osrt -o nsys_report --force-overwrite true python nsys_demo.py
# 看(终端,不需要 GUI):
#   nsys stats nsys_report.nsys-rep
# 看时间线(需 Nsight Systems GUI):scp nsys_report.nsys-rep 下来用 Nsight Systems 打开

import torch, torch.nn as nn, numpy as np
import torch.cuda.nvtx as nvtx

m = nn.Sequential(*[nn.Linear(4096, 4096, bias=False) for _ in range(8)]).cuda().bfloat16()

def step():
    nvtx.range_push("1_data_load_CPU_numpy")          # CPU 数据准备:perfetto 看不见,nsys CPU 采样能抓
    arr = np.random.randn(8192, 4096).astype("float32")
    x_cpu = torch.from_numpy(arr).to(torch.bfloat16)
    nvtx.range_pop()

    nvtx.range_push("2_H2D_copy")                      # 阻塞 H2D:nsys 在 memcpy 轨道单独画出来
    x = x_cpu.cuda()
    nvtx.range_pop()

    nvtx.range_push("3_forward")
    y = m(x)
    nvtx.range_pop()

    nvtx.range_push("4_backward")
    y.float().sum().backward()
    for p in m.parameters():
        p.grad = None
    nvtx.range_pop()

# warmup(不太关心)
for _ in range(2):
    step()
torch.cuda.synchronize()

# 测 5 步,每步用 NVTX 包起来
for i in range(5):
    nvtx.range_push(f"step{i}")
    step()
    nvtx.range_pop()
torch.cuda.synchronize()
print("done → 终端看: nsys stats nsys_report.nsys-rep")
