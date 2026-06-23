# mfu_low.py — 构造几种 MFU 低的情况,理解"为什么算力没用满"
#
# 跑: python mfu_low.py
# MFU = (FLOPs/step × steps/s) / 峰值FLOPS。两类杀手:
#   A) GPU 空闲(在等)  B) GPU 忙但没在做大 matmul(小矩阵/访存/MoE)

import torch, torch.nn as nn, time, os
from torch.utils.flop_counter import FlopCounterMode

PEAK = float(os.environ.get("PEAK_FLOPS", 312e12))   # A100 bf16 dense

def measure(step, label, note=""):
    for _ in range(5): step()                          # warmup
    torch.cuda.synchronize()
    with FlopCounterMode(display=False) as fcm:        # 数 FLOPs(只数 matmul/conv/sdpa 这类)
        step()
    flops = fcm.get_total_flops()
    torch.cuda.synchronize(); t = time.time(); N = 20
    for _ in range(N): step()
    torch.cuda.synchronize()
    sps = N / (time.time() - t)
    achieved = flops * sps
    print(f"  {label:22s} FLOPs/step={flops/1e12:5.2f}T  steps/s={sps:6.1f}  实测={achieved/1e12:6.1f}T  MFU={achieved/PEAK*100:5.1f}%   {note}")

def zero(params):
    for p in params: p.grad = None

# ── ① dense 大矩阵 — 参考(高 MFU)──
def dense_big():
    m = nn.Sequential(*[nn.Linear(4096, 4096, bias=False) for _ in range(8)]).cuda().bfloat16()
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    def step(): m(x).float().sum().backward(); zero(m.parameters())
    measure(step, "① dense 大矩阵", "参考:GEMM 大,喂满张量核心")
    del m, x; torch.cuda.empty_cache()

# ── ② 小矩阵 — GEMM 太小,张量核心喂不满 + 发射开销占比大 ──
def small_matmul():
    m = nn.Sequential(*[nn.Linear(256, 256, bias=False) for _ in range(8)]).cuda().bfloat16()
    x = torch.randn(8192, 256, device="cuda", dtype=torch.bfloat16)
    def step(): m(x).float().sum().backward(); zero(m.parameters())
    measure(step, "② 小矩阵 dim=256", "GEMM 太小→吃不满+发射开销占比大")
    del m, x; torch.cuda.empty_cache()

# ── ③ 访存瓶颈 — 全 elementwise,几乎没 matmul ──
def memory_bound():
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    def step():
        y = x
        for _ in range(40):
            y = torch.nn.functional.gelu(y) * 1.0001 + 0.001   # 全访存型 op,不走张量核心
        y.float().sum().backward(); x.grad = None
    measure(step, "③ 访存瓶颈(elementwise)", "几乎没矩阵乘→MFU≈0(GPU忙在搬数据)")
    del x; torch.cuda.empty_cache()

# ── ④ MoE — 路由 gather/scatter + python 循环 + 每专家小 batch ──
def moe():
    dim, n_exp, topk, T = 2048, 8, 2, 8192
    router = nn.Linear(dim, n_exp).cuda().bfloat16()
    experts = nn.ModuleList([
        nn.Sequential(nn.Linear(dim, 4*dim, bias=False), nn.GELU(), nn.Linear(4*dim, dim, bias=False))
        for _ in range(n_exp)]).cuda().bfloat16()
    x = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16)
    params = list(router.parameters()) + list(experts.parameters())
    def step():
        topi = router(x).softmax(-1).topk(topk, dim=-1).indices     # [T, topk] 路由
        out = torch.zeros_like(x)
        for e in range(n_exp):                                      # python 循环 8 个专家
            idx = (topi == e).any(-1).nonzero(as_tuple=True)[0]
            if idx.numel():
                ye = experts[e](x[idx])                             # gather + 小 batch GEMM
                out = out.index_add(0, idx, ye)                     # scatter
        out.float().sum().backward(); zero(params)
    measure(step, "④ MoE(naive 路由)", "gather/scatter+循环+小batch→掉")
    del router, experts, x; torch.cuda.empty_cache()

# ── ⑤ GPU 空闲 — CPU/数据卡住,GPU 干等(对照 timeline 的 cpu_stall)──
def gpu_idle():
    m = nn.Sequential(*[nn.Linear(4096, 4096, bias=False) for _ in range(8)]).cuda().bfloat16()
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    def step():
        m(x).float().sum().backward(); zero(m.parameters())
        torch.cuda.synchronize(); time.sleep(0.1)      # 模拟 CPU/数据瓶颈,GPU 空等
    measure(step, "⑤ GPU 空闲(CPU卡)", "FLOPs没变,steps/s被拖垮→MFU崩")
    del m, x; torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"MFU = (FLOPs/step × steps/s) / 峰值{PEAK/1e12:.0f}T   [A100 bf16]\n")
    dense_big()
    small_matmul()
    memory_bound()
    moe()
    gpu_idle()
    print("\n两类杀手: A) GPU空闲(⑤) B) GPU忙但没做大matmul(②小/③访存/④MoE)")
    print("→ 低 MFU 怎么查: 先 run_mfu 看数,再 perfetto(空洞=A) / 看 kernel(碎=B)")
