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

print("\n=== (D) FULL TRAINING PATH: ClassifierModule(head_mode=hl_gauss) train+val steps ===")
from src.model.classifier_module import ClassifierModule
mod = ClassifierModule(network=EdgeCrossAttn(pos_fourier=True, use_edge_embed=True, edge_self_attn=True,
                                             budget_cond=True, max_budget=3, value_bins=BINS),
                       head_mode="hl_gauss", bce_reachable_only=True)
batch = {"context": x, "f_labels": y, "r_mask": mask, "loss_mask": mask,
         "contact_px": cpx, "H": Hbud, "ratio": torch.full((B,), 0.5)}
tl = mod.training_step(batch, 0)
tl.backward()
assert torch.isfinite(tl), tl
print(f"  training_step loss = {tl.item():.4f}, backward OK")
with torch.no_grad():
    vl = mod.validation_step(batch, 0)
assert torch.isfinite(vl), vl
print(f"  validation_step loss = {vl.item():.4f} (metrics ranked on E[bin] values)")

print("\n=== (E) backward-compat: default ClassifierModule batch path unchanged (no H key) ===")
mod0 = ClassifierModule(network=EdgeCrossAttn(pos_fourier=True, use_edge_embed=True, edge_self_attn=True),
                        bce_reachable_only=True)
batch0 = {"context": x, "f_labels": (y > 0).float(), "r_mask": mask, "loss_mask": mask,
          "contact_px": cpx, "ratio": torch.full((B,), 0.5)}
tl0 = mod0.training_step(batch0, 0)
assert torch.isfinite(tl0), tl0
print(f"  default sigmoid_bce training_step loss = {tl0.item():.4f} — H absent, path unchanged")

print("\n=== (F) scorer_data budget_h flag (H from H5 'H' dataset, else 1) ===")
import tempfile, h5py, numpy as np
from src.data.scorer_data import ScorerH5Dataset
with tempfile.NamedTemporaryFile(suffix=".h5") as tf:
    with h5py.File(tf.name, "w") as f:
        f["ctx"] = np.random.rand(3, 5, 64, 64).astype(np.float32)
        f["f_grid"] = np.random.randint(0, 2, (3, 60, 5)).astype(np.float32)
        f["r_mask"] = np.ones((3, 60, 5), np.float32)
        f["ratio"] = np.full(3, 0.5, np.float32)
        f["contact_px"] = np.random.rand(3, 60, 2).astype(np.float32)
    d_off = ScorerH5Dataset(tf.name, [0, 1, 2])
    assert "H" not in d_off[0], "budget_h off must not emit H"
    d_on = ScorerH5Dataset(tf.name, [0, 1, 2], budget_h=True)
    assert d_on[0]["H"].item() == 1, "no H5 'H' dataset -> default 1"
    d_off._h5.close(); d_on._h5.close()
    with h5py.File(tf.name, "a") as f:
        f["H"] = np.array([1, 2, 2], np.int8)
    d_on2 = ScorerH5Dataset(tf.name, [0, 1, 2], budget_h=True)
    assert [d_on2[k]["H"].item() for k in range(3)] == [1, 2, 2], "per-row H from H5"
    d_on2._h5.close()
print("  budget_h off: no H key | on: H=1 default | on+H5 'H': per-row values — OK")

print("\nSMOKE TEST PASSED ✓")
