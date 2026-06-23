# mem_anomalies.py — 构造"memory_viz 能一眼看出根因"的显存异常,每个 case 导出一个 snapshot
#
# 跑: python mem_anomalies.py
# 然后本地 scp 下所有 mem_*.pickle,逐个拖进 https://pytorch.org/memory_viz
#
# 关键: memory_viz 的价值 = ① 时间线看显存怎么涨怎么跌 ② 悬停每块看"哪行代码分配的"(栈)
#       → 泄漏(楼梯)、激活峰值、优化器常驻、碎片,这四种长相各不相同。

import torch, torch.nn as nn

def banner(s): print(f"\n{'='*60}\n{s}\n{'='*60}")

def rec_start():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    # 先开录制,带 python 栈 → 悬停能看到真实代码行
    torch.cuda.memory._record_memory_history(max_entries=200000, context="all", stacks="python")

def rec_dump(name):
    torch.cuda.memory._dump_snapshot(f"mem_{name}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)   # 停止录制
    print(f"  导出 mem_{name}.pickle")

def mlp(depth, dim=2048):
    return nn.Sequential(*[nn.Linear(dim, dim) for _ in range(depth)]).cuda().bfloat16()


# ── ① 泄漏: 每步把激活塞进 list,永不释放 → 时间线"楼梯式"上涨 ──
def case_leak():
    banner("① 内存泄漏 — memory_viz 时间线呈【楼梯式上涨,永不回落】")
    rec_start()
    m = mlp(6, dim=4096)
    buf = []
    for i in range(8):
        x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
        y = m(x)
        buf.append(y.detach())   # ← 泄漏:持有引用永不释放(真实版: total_loss += loss 忘了 .item())
        y.float().sum().backward()
    rec_dump("leak")
    del m, buf; torch.cuda.empty_cache()
    print("  看: 总量一级一级往上爬、永不回落(每步 +64MB);")
    print("    悬停那一摞累积的块 → 栈全指向同一行 → 这就是泄漏点")
    print("  根因: 训练循环攒了带引用的 tensor → 修: 存 .item() / .detach().cpu(),或干脆别存")


# ── ② 激活峰值: 前向后、反向前 dump → 抓住激活那一大坨 ──
def case_activation():
    banner("② 激活峰值 — 反向【前】dump,看见前向堆起来的一大坨激活")
    rec_start()
    m = mlp(12, dim=4096)
    x = torch.randn(16384, 4096, device="cuda", dtype=torch.bfloat16)
    y = m(x)                     # 前向:每层激活都被存下来等反向用
    rec_dump("activation_peak")  # ← 反向之前 dump,激活还在
    y.float().sum().backward()
    del m, x, y; torch.cuda.empty_cache()
    print("  看: 一大片 forward 栈的块(每层一个激活)= 反向前的峰值;")
    print("    对比步间静息快照——那时激活已被反向释放,所以之前你看不到它")
    print("  根因: 激活占峰值 → 修: gradient checkpointing(算力换显存)/ 减 batch·seq")


# ── ③ 优化器常驻: AdamW 比 SGD 多 2 份状态 → 多一层永久 slab ──
def case_optimizer():
    banner("③ 优化器状态 — AdamW 的 exp_avg/exp_avg_sq 是【常驻 slab】")
    rec_start()
    m = mlp(8, dim=2048)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        x = torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16)
        m(x).float().sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    rec_dump("optimizer")
    del m, opt; torch.cuda.empty_cache()
    print("  看: 第一次 opt.step() 后冒出一层【永不释放】的 slab;")
    print("    悬停 → 栈指向 adamw → 那是 exp_avg + exp_avg_sq(各等于一整份参数大小)")
    print("  根因: AdamW 每参数 = 1参+1梯+2状态 → 修: 8-bit Adam / SGD / optimizer offload")


# ── ④ 碎片: 变长 shape → reserved 远大于 allocated ──
def case_fragment():
    banner("④ 碎片 — reserved 远大于 allocated(变长 shape 撑出一堆半空 segment)")
    rec_start()
    m = mlp(4, dim=2048)
    keep = []
    for i in range(24):
        S = [512, 8192, 1024, 16384, 2048][i % 5]   # 变长序列 → 分配器反复要不同大小
        x = torch.randn(S, 2048, device="cuda", dtype=torch.bfloat16)
        y = m(x)
        if i % 4 == 0:
            keep.append(y.detach())                 # 留几个,在 segment 里戳出空洞
    rec_dump("fragment")
    a = torch.cuda.memory_allocated()/1e9
    r = torch.cuda.memory_reserved()/1e9
    print(f"  allocated {a:.2f} GB | reserved {r:.2f} GB | 比值 {r/max(a,0.01):.1f}x")
    del m, keep; torch.cuda.empty_cache()
    print("  看: memory_viz 切到 'Allocator State History' / 段视图 →")
    print("    一堆 segment 只用了一部分、大片灰色空闲 = 碎片(reserved 占着却用不上)")
    print("  (本卡很空时比值可能不夸张;线上显存紧张才致命) 修: expandable_segments / 固定 shape / packing")


if __name__ == "__main__":
    case_leak()
    case_activation()
    case_optimizer()
    case_fragment()
    print("\n" + "="*60)
    print("本地 scp 下所有 mem_*.pickle,逐个拖进 https://pytorch.org/memory_viz:")
    print("  leak=楼梯上涨 | activation_peak=一大坨激活 | optimizer=多一层slab | fragment=段里大片空闲")
    print("="*60)
