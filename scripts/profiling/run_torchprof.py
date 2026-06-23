# run_torchprof.py — ③ op 级热点(torch.profiler,纯 torch)
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from mini import make, step

m, x, opt = make()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3),
    record_shapes=True, profile_memory=True, with_stack=True,
) as prof:
    for _ in range(7):
        step(m, x, opt)
        prof.step()

# 终端表:CUDA time 排前的就是热点
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# 时间线:trace.json → ui.perfetto.dev 或 chrome://tracing
prof.export_chrome_trace("trace.json")
print("\n已导出 trace.json → 用 ui.perfetto.dev 看时间线(GPU 空洞/op 间隙)")
