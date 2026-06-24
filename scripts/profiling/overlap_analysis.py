# overlap_analysis.py — 从 nsys sqlite 量化 compute↔memcpy overlap(不用 GUI)
#
# 用法: python overlap_analysis.py wan_offload_nvtx.sqlite
#   (先 nsys stats <report>.nsys-rep 会生成同名 .sqlite)
#
# 算两个指标:
#   ① memcpy 有多少被 compute 掩盖(overlap%)—— 高=offload 传输藏得好
#   ② GPU 计算流空闲%(kernel 之间的缝)—— 高=compute 在等(传输暴露/喂不饱)

import sys, sqlite3, bisect

db = sys.argv[1] if len(sys.argv) > 1 else "wan_offload_nvtx.sqlite"
con = sqlite3.connect(db)
cur = con.cursor()

def intervals(table):
    try:
        return [(s, e) for s, e in cur.execute(f"SELECT start, end FROM {table}") if e > s]
    except sqlite3.OperationalError:
        return []

kern = intervals("CUPTI_ACTIVITY_KIND_KERNEL")
memcpy = intervals("CUPTI_ACTIVITY_KIND_MEMCPY")
print(f"kernels: {len(kern)} | memcpys: {len(memcpy)}")
if not kern:
    print("没读到 kernel 数据(表名可能不同)。可用 `nsys stats` 看 memcpy/kernel summary。")
    sys.exit()

# 合并 kernel 区间 → GPU 计算"忙"的并集
kern.sort()
busy = []
cs, ce = kern[0]
for s, e in kern[1:]:
    if s > ce:
        busy.append((cs, ce)); cs, ce = s, e
    else:
        ce = max(ce, e)
busy.append((cs, ce))

# 指标②:GPU 计算空闲
span = busy[-1][1] - busy[0][0]
busy_time = sum(e - s for s, e in busy)
print(f"\n② GPU 计算: 忙 {busy_time/1e6:.0f}ms / 跨度 {span/1e6:.0f}ms → 空闲 {100*(1-busy_time/span):.0f}%")

# 指标①:memcpy 与 compute 的重叠
if memcpy:
    ends = [b[1] for b in busy]
    total_mem = overlapped = 0
    for ms, me in memcpy:
        total_mem += me - ms
        k = bisect.bisect_right(ends, ms)         # 第一个 end>ms 的 busy 区间
        while k < len(busy) and busy[k][0] < me:  # 往后扫,直到 busy.start >= me
            ov = min(me, busy[k][1]) - max(ms, busy[k][0])
            if ov > 0: overlapped += ov
            k += 1
    print(f"\n① memcpy 总 {total_mem/1e6:.0f}ms | 被 compute 掩盖 {overlapped/1e6:.0f}ms "
          f"({100*overlapped/total_mem:.0f}%) | 暴露 {(total_mem-overlapped)/1e6:.0f}ms "
          f"({100*(1-overlapped/total_mem):.0f}%)")
    print("\n→ overlap% 高 + 空闲% 低 = offload 传输被计算掩盖得好(几乎不掉速)")
    print("→ overlap% 低 + 空闲% 高 = 传输暴露在关键路径,compute 在等传输")
