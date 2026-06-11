"""Smoke test: budget-conditioned horizon-Q (EdgeCrossAttn + H-embedding + HL-Gauss head).

Verifies (CPU): (1) the new budget_cond + value_bins flags forward to the right shapes; (2) HL-Gauss
target/loss/value are sane; (3) backward flows grads through the H-embedding + value head; (4) the
DEFAULT EdgeCrossAttn (flags off) is unchanged (binary 60x5 logits, no H). Run:
  /scratch/dm1487/envs/namo/bin/python scripts/smoke_budget_q.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.model.dit.edge_crossattn import EdgeCrossAttn
from src.model.hl_gauss import HLGauss

torch.manual_seed(0)
B, ND, NE, BINS = 4, 5, 60, 51
x = torch.randn(B, 5, 64, 64)
cpx = torch.rand(B, NE, 2) * 64
Hbud = torch.randint(1, 3, (B,))                       # remaining budget in {1,2}

print("=== (A) budget-Q: budget_cond=True, value_bins=51, champion flags ===")
m = EdgeCrossAttn(pos_fourier=True, use_edge_embed=True, edge_self_attn=True,
                  budget_cond=True, max_budget=3, value_bins=BINS)
out = m(x, cpx, H=Hbud)
assert out.shape == (B, NE, ND, BINS), out.shape
print(f"  forward OK -> {tuple(out.shape)} (B,60,depths,bins)")

hl = HLGauss(num_bins=BINS)
# gamma targets in {0, 0.9, 1.0}; masked ~30/300 cells per scene (PU sampling)
y = torch.zeros(B, NE, ND)
y[torch.rand_like(y) > 0.6] = 0.9
y[torch.rand_like(y) > 0.85] = 1.0
mask = (torch.rand(B, NE, ND) < 0.10).float()         # ~30/300 sampled
loss = hl.loss(out, y, mask)
loss.backward()
print(f"  HL-Gauss masked loss = {loss.item():.4f}  (finite={torch.isfinite(loss).item()})")
# grads flow to H-embedding and value head?
g_bud = m.budget_embed.weight.grad
g_head = m.head[-1].weight.grad
assert g_bud is not None and g_bud.abs().sum() > 0, "no grad to budget_embed"
assert g_head is not None and g_head.abs().sum() > 0, "no grad to head"
print("  grads flow to budget_embed AND value head: OK")
val = hl.value(out)                                   # (B,60,5) expected value in [0,1]
assert val.shape == (B, NE, ND) and (val >= 0).all() and (val <= 1).all(), (val.min(), val.max())
print(f"  value map -> {tuple(val.shape)} in [{val.min():.3f},{val.max():.3f}]")
# policy = top-k, V(s) = top-k-mean pool (NOT max, per H0b)
flat = val.reshape(B, -1)
topk = flat.topk(5, dim=-1).values
print(f"  V(s)=top5-mean per scene: {[round(v,3) for v in topk.mean(-1).tolist()]}")

print("\n=== (B) backward-compat: default EdgeCrossAttn (flags off) unchanged ===")
m0 = EdgeCrossAttn(pos_fourier=True, use_edge_embed=True, edge_self_attn=True)  # no budget, no bins
out0 = m0(x, cpx)                                      # no H arg
assert out0.shape == (B, NE, ND), out0.shape
print(f"  forward OK (no H) -> {tuple(out0.shape)} (B,60,depths) binary logits — unchanged")

print("\n=== (C) HL-Gauss sanity: target/value round-trip ===")
for v in [0.0, 0.9, 1.0]:
    t = hl.target(torch.tensor(v))
    recon = (t * hl.bin_centers).sum().item()
    print(f"  y={v:.2f} -> soft-hist E[bin]={recon:.4f}")

print("\nSMOKE TEST PASSED ✓")
