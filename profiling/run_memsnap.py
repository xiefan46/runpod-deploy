# run_memsnap.py — ② 显存快照(定位 OOM / 看显存被谁占,纯 torch)
import torch
from mini import make, step

m, x, opt = make()

torch.cuda.memory._record_memory_history(max_entries=200000)
for _ in range(3):
    step(m, x, opt)
torch.cuda.memory._dump_snapshot("snap.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # 停止记录

print("已导出 snap.pickle")
print("→ 上传到 https://pytorch.org/memory_viz 看:")
print("   每块显存谁分配的、时间线、峰值来自哪(参数/激活/优化器)")
print("提示:真实场景在 OOM 前 dump 一次,就知道是谁撑爆的")
