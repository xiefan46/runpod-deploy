# timeline_anomalies.py — 构造"时间线上能看到空洞"的异常,导出多个 trace 供 perfetto 对比
#
# 跑: python timeline_anomalies.py
# 然后本地 scp 下所有 trace_*.json,拖进 ui.perfetto.dev,对比看 GPU stream 上的空洞
#
# 关键: 空洞 = GPU 空闲在等别人(等 CPU / 等传输 / 等发射)。baseline 无空洞;其余三个有。

import torch, torch.nn as nn, time, numpy as np
from torch.profiler import profile, ProfilerActivity, schedule

def prof_run(fn, name, steps=6):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=1, active=3)) as p:
        for _ in range(steps):
            fn(); p.step()
    out = f"trace_{name}.json"
    p.export_chrome_trace(out)
    print(f"  导出 {out}")

def banner(s): print(f"\n{'='*60}\n{s}\n{'='*60}")

def mlp(depth=8):
    return nn.Sequential(*[nn.Linear(4096, 4096, bias=False) for _ in range(depth)]).cuda().bfloat16()
def zero(m):
    for p in m.parameters(): p.grad = None

# ── baseline: GPU-bound,无空洞(对照组)──
def case_baseline():
    banner("baseline — GPU-bound,时间线应【无空洞】(对照)")
    m = mlp(12); x = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
    def step():
        y = m(x); y.float().sum().backward(); zero(m)
    prof_run(step, "baseline")
    print("  看: CUDA stream 上 kernel 一个接一个,几乎没缝")

# ── ① CPU 卡顿: GPU 等 CPU 数据准备 ──
def case_cpu_stall():
    banner("① CPU 卡顿 — GPU 空闲等 CPU(大空洞)")
    m = mlp(4); x = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
    def step():
        _ = np.random.randn(4_000_000).sum()   # CPU 计算(模拟数据准备)
        time.sleep(0.02)                        # CPU 阻塞
        y = m(x); y.float().sum().backward(); zero(m)
    prof_run(step, "cpu_stall")
    print("  看: 每步 GPU 做完后有一大段空白(CPU 在 numpy+sleep,GPU 干等)")
    print("  根因: 数据/CPU 瓶颈 → 修: 预计算/多 worker/预取/overlap")

# ── ② 发射开销 bound: 一堆 tiny kernel,GPU 喂不饱 ──
def case_launch_bound():
    banner("② launch-bound — 大量微小 kernel,GPU 大半时间空闲")
    xs = [torch.randn(64, 64, device="cuda") for _ in range(400)]
    def step():
        s = xs[0]
        for t in xs:
            s = s + t * 1.0001     # 几百个微小 kernel
        s.sum().item()             # 同步
    prof_run(step, "launch_bound")
    print("  看: 一堆极短的 kernel,之间有空隙(CPU 来不及发射)→ GPU 利用率极低")
    print("  根因: 太多小 op → 修: torch.compile / cudagraph / 算子融合")

# ── ③ H2D 传输 bound: 每步从 CPU 拷数据上 GPU,传输时 GPU 空闲 ──
def case_h2d():
    banner("③ H2D 传输 — 每步阻塞拷数据上 GPU,传输时 GPU 空闲")
    m = mlp(4)
    def step():
        x_cpu = torch.randn(32, 4096, dtype=torch.bfloat16)  # 在 CPU 上造
        x = x_cpu.cuda()                                     # 阻塞 H2D 传输
        y = m(x); y.float().sum().backward(); zero(m)
    prof_run(step, "h2d_copy")
    print("  看: 每步开头有 Memcpy HtoD 条,期间 GPU 计算流空闲")
    print("  根因: 同步 H2D → 修: pin_memory + non_blocking=True + 预取 overlap")

if __name__ == "__main__":
    case_baseline()
    case_cpu_stall()
    case_launch_bound()
    case_h2d()
    print("\n" + "="*60)
    print("本地 scp 下所有 trace,拖进 ui.perfetto.dev 对比:")
    print("  baseline 无空洞 | cpu_stall/h2d 有大空洞 | launch_bound 一堆碎 kernel")
    print("="*60)
