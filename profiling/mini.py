# mini.py — 一个最小 transformer stack,用来学 profiling 工具(不碰框架复杂度)
import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Mini(nn.Module):
    def __init__(self, dim=2048, heads=16, depth=12):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

def make(B=4, S=2048, dim=2048, heads=16, depth=12):
    m = Mini(dim, heads, depth).cuda().bfloat16()
    x = torch.randn(B, S, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    return m, x, opt

def step(m, x, opt):
    opt.zero_grad(set_to_none=True)
    out = m(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    opt.step()
    return loss
