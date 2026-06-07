"""
Unit test for soft Gaussian edge labels in ClassifierModule.

Tests:
  (a) sigma=0 → soft_target == f_grid exactly (identity / flag-off)
  (b) sigma=1 (edge), sigma=0.7 (depth):
      - positive stays 1.0
      - same-face arc-neighbors at distance 1 get ~exp(-0.5) ≈ 0.6065
      - edges on OTHER faces stay 0
      - adjacent depths get partial credit from depth Gaussian
      - no target exceeds 1.0

Edge layout reminder:
  60 edges, 4 faces × 15 pts, interleaved parity.
    even e <30: top face,    j = e//2
    odd  e <30: bottom face, j = e//2
    even e>=30: right face,  j = (e-30)//2
    odd  e>=30: left face,   j = (e-30)//2
  Same-face neighbor of e = e±2 (same parity, same half [0..29] or [30..59]).

We place a single positive at edge=10 (top face, j=5), depth=2.
  Same-face neighbors: edge=8 (j=4, dist=1), edge=12 (j=6, dist=1), and further.
  Cross-face edges (odd ones like 11, or e>=30) must stay 0.
"""
import sys
sys.path.insert(0, "/cache/home/dm1487/projects/namo/sage_learning")

import torch
import numpy as np
from unittest.mock import MagicMock

# Patch network to a dummy to avoid instantiating full model
import torch.nn as nn

from src.model.classifier_module import ClassifierModule


def make_module(soft_edge_sigma: float, soft_depth_sigma: float) -> ClassifierModule:
    net = nn.Linear(1, 1)  # dummy
    return ClassifierModule(
        network=net,
        soft_edge_sigma=soft_edge_sigma,
        soft_depth_sigma=soft_depth_sigma,
    )


def make_f_grid(pos_edge: int, pos_depth: int, B: int = 1, E: int = 60, D: int = 5) -> torch.Tensor:
    f = torch.zeros(B, E, D)
    f[:, pos_edge, pos_depth] = 1.0
    return f


def test_identity(pos_edge=10, pos_depth=2):
    """sigma=0 → soft_target must equal f_grid exactly."""
    m = make_module(0.0, 0.0)
    f = make_f_grid(pos_edge, pos_depth)
    soft = m._build_soft_target(f)
    assert soft is f, f"Expected identity (same tensor object), got new tensor"
    print("  PASS: sigma=0 returns f_grid unchanged (identity object)")


def test_soft_labels(pos_edge=10, pos_depth=2, sigma_e=1.0, sigma_d=0.7):
    """sigma>0 → spread correctly, positives preserved, cross-face=0, no value >1."""
    m = make_module(sigma_e, sigma_d)
    f = make_f_grid(pos_edge, pos_depth, B=2)  # test with batch of 2
    soft = m._build_soft_target(f)

    # ---- (1) positives must stay exactly 1.0 ----
    for b in range(2):
        v = soft[b, pos_edge, pos_depth].item()
        assert abs(v - 1.0) < 1e-6, f"Positive degraded: got {v:.6f}"
    print(f"  PASS: positive at (edge={pos_edge}, depth={pos_depth}) stays 1.0")

    # ---- (2) same-face arc-neighbors at dist=1 get exp(-0.5) ≈ 0.6065 (depth same) ----
    #  depth weight for depth=pos_depth itself: exp(0) = 1.0
    #  edge weight for dist=1: exp(-1/(2*sigma_e^2)) = exp(-0.5) for sigma_e=1.0
    expected_edge_weight = float(np.exp(-1.0 / (2 * sigma_e ** 2)))
    # pos_edge=10 (top face, even): neighbors are 8 and 12 (both top face, even)
    same_face_neighbors = [8, 12]
    for ne in same_face_neighbors:
        v = soft[0, ne, pos_depth].item()
        assert abs(v - expected_edge_weight) < 1e-4, (
            f"Same-face neighbor edge={ne}, depth={pos_depth}: "
            f"expected {expected_edge_weight:.4f}, got {v:.4f}"
        )
    print(f"  PASS: same-face arc-neighbors (edges {same_face_neighbors}) at pos_depth "
          f"get {expected_edge_weight:.4f} ≈ exp(-0.5)")

    # ---- (3) edges on OTHER faces must be 0 ----
    #  pos_edge=10 is top face (even, <30). Bottom face = odd <30. Right/left = >=30.
    other_face_edges = [9, 11, 13, 30, 31, 32, 33]
    for oe in other_face_edges:
        v = soft[0, oe, pos_depth].item()
        assert abs(v) < 1e-6, (
            f"Cross-face edge={oe} should be 0, got {v:.6f}"
        )
    print(f"  PASS: cross-face edges {other_face_edges} all stay 0")

    # ---- (4) adjacent depths get partial credit at the positive edge ----
    #  depth Gaussian: exp(-(Δd)^2 / (2*sigma_d^2))
    for d in range(5):
        delta_d = d - pos_depth
        expected_d_weight = float(np.exp(-(delta_d ** 2) / (2 * sigma_d ** 2)))
        v = soft[0, pos_edge, d].item()
        assert abs(v - expected_d_weight) < 1e-4, (
            f"Depth {d} at pos_edge: expected {expected_d_weight:.4f}, got {v:.4f}"
        )
    print(f"  PASS: adjacent depths at pos_edge get correct Gaussian credit")
    depth_vals = [soft[0, pos_edge, d].item() for d in range(5)]
    print(f"         depth weights: {[f'{v:.4f}' for v in depth_vals]}")

    # ---- (5) no value exceeds 1.0 ----
    max_val = soft.max().item()
    assert max_val <= 1.0 + 1e-6, f"soft_target exceeds 1.0: max={max_val}"
    print(f"  PASS: max(soft_target) = {max_val:.6f} ≤ 1.0")

    # ---- (6) combined: same-face neighbor at dist=1, adjacent depth gets edge*depth weight ----
    ne, d_adj = 8, pos_depth + 1
    expected_combined = expected_edge_weight * float(np.exp(-1.0 / (2 * sigma_d ** 2)))
    v = soft[0, ne, d_adj].item()
    assert abs(v - expected_combined) < 1e-4, (
        f"Neighbor (edge={ne}, depth={d_adj}): expected {expected_combined:.4f}, got {v:.4f}"
    )
    print(f"  PASS: arc-neighbor+adj-depth combined weight = {expected_combined:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test (a): sigma=0 → identity")
    test_identity()

    print()
    print("Test (b): sigma_e=1.0, sigma_d=0.7 → soft labels")
    test_soft_labels(sigma_e=1.0, sigma_d=0.7)

    print()
    print("ALL TESTS PASSED")
