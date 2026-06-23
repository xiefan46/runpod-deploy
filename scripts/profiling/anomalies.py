# anomalies.py — 故意构造异常,学会在 run_basic 的数字里认出它们
#
# 跑: python anomalies.py
# 每个 case 打印:异常表现 + 该用哪个脚本查根因
# 关键认知: run_basic 是"症状探测器"(慢/OOM/泄漏/碎片/假快),根因要靠 torchprof / memsnap / nsys / ncu

import torch, torch.nn as nn, time, random
from mini import Mini, make, step

def measure(m, x, opt, N=15, sync=True):
    for _ in range(5):
        step(m, x, opt)
    if sync:
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        step(m, x, opt)
    if sync:
        torch.cuda.synchronize()
    return (time.time() - t) / N

def banner(s):
    print(f"\n{'='*64}\n{s}\n{'='*64}")

# ── ① 不 sync = 计时假快 ──
def case_no_sync():
    banner("① 不 sync → 计时假快(throughput 高到不真实)")
    m, x, opt = make()
    dt_t = measure(m, x, opt, sync=True)
    dt_f = measure(m, x, opt, sync=False)
    print(f"  正确(sync)   : {dt_t*1000:6.1f} ms/step")
    print(f"  错误(no sync): {dt_f*1000:6.1f} ms/step  ← 假!只测了'入队'时间")
    print("  症状: throughput 高到不真实 | 根因: 忘了 torch.cuda.synchronize()")
    del m, x, opt; torch.cuda.empty_cache()

# ── ② 内存泄漏 ──
def case_leak():
    banner("② 内存泄漏 → current allocated 每步上涨")
    m, x, opt = make(B=4, S=2048)
    leak = []
    base = None
    for i in range(12):
        out = m(x); loss = out.float().pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        leak.append(out.detach())  # ← 持有引用 = 泄漏
        cur = torch.cuda.memory_allocated()/1e9
        if base is None: base = cur
        if i % 3 == 0:
            print(f"  step {i:2d}: current alloc = {cur:.2f} GB  (持续涨 = 泄漏)")
    print(f"  净增 ≈ {torch.cuda.memory_allocated()/1e9 - base:.2f} GB")
    print("  症状: current alloc 单调涨; Tot Freed << Tot Alloc; 久了 OOM")
    print("  根因定位: run_memsnap.py(看谁在累积)")
    del m, x, opt, leak; torch.cuda.empty_cache()

# ── ③ 碎片 ──
def case_fragmentation():
    banner("③ 碎片 → reserved 明显大于 allocated(变长 shape 触发)")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = Mini(2048).cuda().bfloat16()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for i in range(25):
        S = random.choice([256, 2048, 512, 3072, 768, 4096])  # 变长序列 → 碎片
        x = torch.randn(4, S, 2048, device="cuda", dtype=torch.bfloat16)
        out = m(x); out.float().pow(2).mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    alloc = torch.cuda.memory_allocated()/1e9
    reserved = torch.cuda.memory_reserved()/1e9
    retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    print(f"  current alloc: {alloc:.2f} GB | reserved: {reserved:.2f} GB | ratio: {reserved/max(alloc,0.01):.1f}x | retries: {retries}")
    print("  症状: reserved/alloc 比值高;显存紧张时 retries>0(本卡空,retries 可能仍为0)")
    print("  根因: 变长序列/变 batch → 分配器反复申请不同大小 segment")
    print("  缓解: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 或固定 shape / sequence packing")
    del m, opt; torch.cuda.empty_cache()

# ── ④ CPU/sync 卡顿(run_basic 看不出根因)──
def case_cpu_stall():
    banner("④ CPU/sync 卡顿 → throughput 掉,但 run_basic 说不出为什么")
    m, x, opt = make()
    dt_ok = measure(m, x, opt)
    def slow(m, x, opt):
        opt.zero_grad(set_to_none=True)
        loss = m(x).float().pow(2).mean()
        loss.backward(); opt.step()
        _ = loss.item()   # 每步强制 CPU-GPU 同步
        time.sleep(0.05)  # 模拟 CPU 数据准备瓶颈
    for _ in range(5): slow(m, x, opt)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(15): slow(m, x, opt)
    torch.cuda.synchronize(); dt_bad = (time.time()-t)/15
    print(f"  正常      : {dt_ok*1000:6.1f} ms/step")
    print(f"  CPU 卡顿  : {dt_bad*1000:6.1f} ms/step  ← 慢了")
    print("  症状: throughput 低 —— 但 run_basic 只告诉你'慢',说不出原因")
    print("  根因定位: run_torchprof.py 时间线(会看到 GPU 空洞 = 在等 CPU)→ 这就是为什么需要 profiler")
    del m, x, opt; torch.cuda.empty_cache()

# ── ⑤ 优化器选择 → 常驻显存差很多(教育性)──
def case_optimizer_mem():
    banner("⑤ 优化器选择 → 常驻显存差很多")
    for name, mk in [("SGD  ", lambda p: torch.optim.SGD(p, lr=0.1)),
                     ("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-4))]:
        torch.cuda.empty_cache()
        m = Mini(2048).cuda().bfloat16()
        x = torch.randn(2, 512, 2048, device="cuda", dtype=torch.bfloat16)
        opt = mk(m.parameters())
        for _ in range(3):
            opt.zero_grad(set_to_none=True); m(x).float().pow(2).mean().backward(); opt.step()
        print(f"  {name}: current alloc = {torch.cuda.memory_allocated()/1e9:.2f} GB")
        del m, x, opt
    print("  AdamW 比 SGD 多 2 份状态(exp_avg/exp_avg_sq)→ 常驻更大")
    print("  省显存: 8-bit Adam / SGD / optimizer offload")
    torch.cuda.empty_cache()

# ── ⑥ OOM(放最后,捕获)──
def case_oom():
    banner("⑥ OOM → 故意撑爆(捕获错误)")
    try:
        m = Mini(4096, 32, 24).cuda().bfloat16()
        x = torch.randn(8, 8192, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        m(x).float().pow(2).mean().backward(); opt.step()
        print("  没 OOM(这卡够大),继续调大 dim/seq/batch 即可触发")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✅ 触发 OOM: {str(e).splitlines()[0][:110]}")
        print("  根因定位: run_memsnap.py(OOM 前 dump,看谁吃满)")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    case_no_sync()
    case_leak()
    case_fragmentation()
    case_cpu_stall()
    case_optimizer_mem()
    case_oom()
    print("\n" + "="*64)
    print("小结: run_basic 是'症状探测器';根因要 run_torchprof(慢)/ run_memsnap(显存)")
    print("="*64)
