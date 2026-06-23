# ncu_demo.py — 给 ncu 两个对比 kernel,看 roofline 的两条边
#   GEMM    → compute-bound(算术强度高,贴"算力顶")
#   gelu    → memory-bound(每读1字节只算几次,贴"带宽顶")
#
# 跑(终端,看 Compute% vs Memory%,最直观):
#   ncu --section SpeedOfLight -k "regex:gemm|elementwise" -c 2 python ncu_demo.py
# 存报告给 GUI 看 roofline 图:
#   ncu --set full -k "regex:gemm|elementwise" -c 2 -f -o ncu_report python ncu_demo.py
#
# 注:ncu 要"重放"每个 kernel 采硬件计数器,所以只 profile 少数几个(-c 限制)。
#     若报 ERR_NVGPUCTRPERM = 宿主机锁了性能计数器(容器内改不了),把错误发我。

import torch

a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
x = torch.randn(128_000_000, device="cuda", dtype=torch.bfloat16)

c = a @ b                          # ① compute-bound: 大矩阵乘
y = torch.nn.functional.gelu(x)    # ② memory-bound: 逐元素
torch.cuda.synchronize()
print("done")
