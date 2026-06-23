# run_mfu.py — ① 算 MFU(用 PyTorch 内置 FlopCounterMode,无需外部库)
import torch, time
from torch.utils.flop_counter import FlopCounterMode
from mini import make, step

m, x, opt = make()

# 1) 数一个 step 的 FLOPs(含 fwd+bwd,因为 step 里 backward 了)
with FlopCounterMode(display=False) as fcm:
    step(m, x, opt)
flops = fcm.get_total_flops()

# 2) 测 steps/sec
for _ in range(5):
    step(m, x, opt)
torch.cuda.synchronize()
t = time.time(); N = 20
for _ in range(N):
    step(m, x, opt)
torch.cuda.synchronize()
sps = N / (time.time() - t)

# 3) MFU = 实际算力 / 峰值算力
# 峰值(bf16 dense,查你的卡): A100 ~312e12, H100 ~990e12, B200/B300 更高
PEAK = float(__import__("os").environ.get("PEAK_FLOPS", 312e12))
mfu = flops * sps / PEAK * 100

print(f"FLOPs/step : {flops/1e12:.2f} TFLOP (fwd+bwd)")
print(f"steps/s    : {sps:.2f}")
print(f"peak FLOPS : {PEAK/1e12:.0f} TFLOPS  (用 PEAK_FLOPS 环境变量改)")
print(f"MFU        : {mfu:.1f}%   (20% 空间大 / 40%+ 算好)")
