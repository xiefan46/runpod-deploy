# run_basic.py — ① 测吞吐 + 峰值显存(纯 torch,零额外安装)
import torch, time
from mini import make, step

m, x, opt = make()

for _ in range(5):  # warmup
    step(m, x, opt)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

t = time.time(); N = 20
for _ in range(N):
    step(m, x, opt)
torch.cuda.synchronize()
dt = (time.time() - t) / N

print(f"step time   : {dt*1000:.1f} ms")
print(f"throughput  : {x.shape[0]/dt:.1f} samples/s")
print(f"peak alloc  : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"peak reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB  (远大于 alloc = 碎片)")
print()
print(torch.cuda.memory_summary())
