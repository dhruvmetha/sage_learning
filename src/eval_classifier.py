#!/usr/bin/env python3
"""
Evaluate primitive feasibility classifier against ground truth F grids.

Computes:
- Top-k accuracy by difficulty level
- Face prediction accuracy
- Contact point precision
- Generates comparison heatmaps (ground truth vs predicted)
- Random baseline comparison

Usage:
    python src/eval_classifier.py \
        --checkpoint /path/to/checkpoints/best.ckpt \
        --data-dir /common/users/dm1487/namo_data/f_characterization/classifier_test_npz \
        --output-dir /path/to/eval_results
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from src.data.classifier_data import ClassifierDataset, ClassifierDataModule
from src.model.classifier_module import ClassifierModule, PrimitiveClassifierCNN
from src.model.dit.dit_classifier import DiTClassifier
from torchvision import transforms


# ── Constants ────────────────────────────────────────────────────────────

FACES_EDGES = [
    list(range(0, 30, 2)),   # Top (push down)
    list(range(1, 30, 2)),   # Bottom (push up)
    list(range(30, 60, 2)),  # Right (push left)
    list(range(31, 60, 2)),  # Left (push right)
]
FACE_NAMES = ["Top", "Bottom", "Right", "Left"]


def classify_difficulty(ratio):
    if ratio < 0.05:
        return "very_hard"
    elif ratio < 0.15:
        return "hard"
    elif ratio < 0.40:
        return "medium"
    elif ratio < 0.70:
        return "easy"
    else:
        return "very_easy"


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(model, dataloader, device):
    """Run model on all data, collect predictions and ground truth."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            f_labels = batch['f_labels']       # (B, 60, 10)
            r_mask = batch['r_mask']            # (B, 60, 10) exact per-depth reachability
            cp_reachable = batch['cp_reachable']  # (B, 60, 10) contact-point-level (inference)
            F_count = batch['F']               # (B,)
            R_count = batch['R']               # (B,)
            ratio = batch['ratio']             # (B,)

            logits = model(context)            # (B, 60, 10)
            scores = torch.sigmoid(logits).cpu()

            for i in range(context.shape[0]):
                results.append({
                    'scores': scores[i].numpy(),           # (60, 10)
                    'labels': f_labels[i].numpy(),         # (60, 10)
                    'r_mask': r_mask[i].numpy(),           # (60, 10) exact
                    'cp_reachable': cp_reachable[i].numpy(),  # (60, 10) inference-level
                    'F': int(F_count[i]),
                    'R': int(R_count[i]),
                    'ratio': float(ratio[i]),
                })

    return results


def compute_topk_accuracy(results, k_values=[1, 3, 5, 10, 20]):
    """Compute top-k accuracy under realistic inference conditions.

    At inference, we only know contact-point-level reachability (from wavefront).
    We do NOT know depth reachability — so the model may pick a primitive whose
    contact point is reachable but whose depth is blocked during push execution.
    That counts as a wasted attempt (not a hit, not skipped).
    """
    # Two modes: oracle (exact r_mask) and realistic (cp_reachable)
    oracle_acc = defaultdict(lambda: {k: [] for k in k_values})
    realistic_acc = defaultdict(lambda: {k: [] for k in k_values})
    wasted_by_diff = defaultdict(list)  # how many of top-k are depth-unreachable

    for r in results:
        exact_mask = r['r_mask'].astype(bool)       # what's actually reachable
        cp_mask = r['cp_reachable'].astype(bool)     # what we know at inference
        labels = r['labels']
        scores = r['scores']
        diff = classify_difficulty(r['ratio'])

        if labels.sum() == 0:
            continue

        # Oracle: mask with exact reachability (best case)
        oracle_scores = scores[exact_mask]
        oracle_labels = labels[exact_mask]
        if len(oracle_scores) > 0 and oracle_labels.sum() > 0:
            for k in k_values:
                kk = min(k, len(oracle_scores))
                top_k_idx = np.argsort(oracle_scores)[::-1][:kk]
                hit = oracle_labels[top_k_idx].sum() > 0
                oracle_acc[diff][k].append(float(hit))

        # Realistic: mask with contact-point-level reachability
        # The model picks from all depths at reachable contact points
        # Some of those depths may not be actually reachable (wasted attempt)
        cp_scores = scores.copy()
        cp_scores[~cp_mask] = -1  # zero out unreachable contact points

        flat_scores = cp_scores.flatten()
        flat_labels = labels.flatten()
        flat_exact = exact_mask.flatten()

        # Sort by score descending
        ranked = np.argsort(flat_scores)[::-1]
        # Filter to only cp-reachable candidates
        ranked = [i for i in ranked if cp_mask.flatten()[i]]

        if not ranked:
            continue

        for k in k_values:
            kk = min(k, len(ranked))
            top_k = ranked[:kk]

            # Count hits (actually feasible)
            hits = sum(flat_labels[i] == 1 for i in top_k)
            # Count wasted (cp reachable but depth blocked — not in exact r_mask)
            wasted = sum(not flat_exact[i] for i in top_k)

            realistic_acc[diff][k].append(float(hits > 0))

        # Track wasted attempts at top-5
        top5 = ranked[:min(5, len(ranked))]
        wasted_5 = sum(not flat_exact[i] for i in top5)
        wasted_by_diff[diff].append(wasted_5)

    return oracle_acc, realistic_acc, wasted_by_diff

    return by_difficulty


def compute_random_baseline(results, k_values=[1, 3, 5, 10, 20]):
    """Random baseline: expected top-k hit rate = 1 - (1 - |F|/|R|)^k."""
    by_difficulty = defaultdict(lambda: {k: [] for k in k_values})

    for r in results:
        if r['R'] == 0 or r['F'] == 0:
            continue
        diff = classify_difficulty(r['ratio'])
        ratio = r['ratio']

        for k in k_values:
            kk = min(k, r['R'])
            # Probability of hitting at least one: 1 - C(R-F, k) / C(R, k)
            # Approximate: 1 - (1 - F/R)^k
            p_hit = 1 - (1 - ratio) ** kk
            by_difficulty[diff][k].append(p_hit)

    return by_difficulty


def compute_face_accuracy(results):
    """Does the classifier's highest-scored primitive land on the correct face?"""
    by_difficulty = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in results:
        mask = r['r_mask'].astype(bool)
        labels = r['labels']
        scores = r['scores']
        diff = classify_difficulty(r['ratio'])

        if labels.sum() == 0:
            continue

        # Ground truth active faces
        true_faces = set()
        for fi, edges in enumerate(FACES_EDGES):
            if labels[edges, :].sum() > 0:
                true_faces.add(fi)

        # Predicted face: face of highest-scored reachable primitive
        reachable_scores = scores.copy()
        reachable_scores[~mask] = -1
        best_idx = np.unravel_index(reachable_scores.argmax(), reachable_scores.shape)
        best_edge = best_idx[0]

        pred_face = None
        for fi, edges in enumerate(FACES_EDGES):
            if best_edge in edges:
                pred_face = fi
                break

        by_difficulty[diff]['total'] += 1
        if pred_face in true_faces:
            by_difficulty[diff]['correct'] += 1

    return by_difficulty


def compute_contact_point_distance(results):
    """Average distance (in contact point indices) between predicted and nearest true CP."""
    by_difficulty = defaultdict(list)

    for r in results:
        mask = r['r_mask'].astype(bool)
        labels = r['labels']
        scores = r['scores']
        diff = classify_difficulty(r['ratio'])

        if labels.sum() == 0:
            continue

        # Best predicted (edge_idx, depth) among reachable
        reachable_scores = scores.copy()
        reachable_scores[~mask] = -1
        pred_edge, pred_depth = np.unravel_index(reachable_scores.argmax(),
                                                  reachable_scores.shape)

        # Find which face the prediction is on
        pred_face = None
        pred_cp = None
        for fi, edges in enumerate(FACES_EDGES):
            if pred_edge in edges:
                pred_face = fi
                pred_cp = edges.index(pred_edge)
                break

        # Find nearest true contact point on the same face
        if pred_face is not None:
            true_cps = []
            for cp_idx, edge in enumerate(FACES_EDGES[pred_face]):
                if labels[edge, :].sum() > 0:
                    true_cps.append(cp_idx)

            if true_cps:
                min_dist = min(abs(pred_cp - tcp) for tcp in true_cps)
                by_difficulty[diff].append(min_dist)
            else:
                # Wrong face — distance = max
                by_difficulty[diff].append(15)

    return by_difficulty


# ── Visualization ────────────────────────────────────────────────────────

def plot_topk_comparison(classifier_acc, random_acc, output_path):
    """Bar chart: classifier vs random baseline top-k accuracy by difficulty."""
    difficulties = ["very_hard", "hard", "medium", "easy", "very_easy"]
    diff_labels = ["Very Hard\n(<5%)", "Hard\n(5-15%)", "Medium\n(15-40%)",
                   "Easy\n(40-70%)", "Very Easy\n(>70%)"]
    k_values = [1, 5, 10]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5))

    for ki, k in enumerate(k_values):
        ax = axes[ki]
        x = np.arange(len(difficulties))
        width = 0.35

        cls_vals = []
        rnd_vals = []
        for diff in difficulties:
            cls_hits = classifier_acc.get(diff, {}).get(k, [])
            rnd_hits = random_acc.get(diff, {}).get(k, [])
            cls_vals.append(np.mean(cls_hits) * 100 if cls_hits else 0)
            rnd_vals.append(np.mean(rnd_hits) * 100 if rnd_hits else 0)

        bars1 = ax.bar(x - width/2, cls_vals, width, label='Classifier',
                        color=colors[ki], alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, rnd_vals, width, label='Random',
                        color='gray', alpha=0.5, edgecolor='black', linewidth=0.5)

        # Value labels
        for bar, val in zip(bars1, cls_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, rnd_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title(f'Top-{k} Accuracy', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(diff_labels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_heatmap_comparison(result, output_path, title=""):
    """Side-by-side: ground truth F grid vs classifier predicted scores.

    Realistic mode: uses contact-point-level reachability (what's known at inference).

    Ground truth colors:
        Green  = feasible (push succeeds, opens passage)
        Gray   = reachable & tried, but didn't clear the passage
        Purple = contact point reachable, but this depth is blocked (depth-unreachable)
                 Classifier would waste an attempt here.
        Black  = contact point not reachable (robot can't reach this face/point)

    Classifier colors:
        Green-Yellow-Red heatmap over all depths at reachable contact points
        Black = contact point not reachable
    """
    labels = result['labels']             # (60, 10)
    scores = result['scores']             # (60, 10)
    r_mask = result['r_mask']             # (60, 10) exact per-depth
    cp_reachable = result['cp_reachable'] # (60, 10) contact-point level

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Ground Truth (left) ──
    ax = axes[0]

    # Build RGBA image directly for fine control
    gt_rgba = np.zeros((60, 10, 4))
    row = 0
    for edges in FACES_EDGES:
        for ei in edges:
            for d in range(10):
                if not cp_reachable[ei, d]:
                    # Contact point not reachable — black
                    gt_rgba[row, d] = [0, 0, 0, 1]
                elif not r_mask[ei, d]:
                    # Contact point reachable but depth blocked — purple
                    gt_rgba[row, d] = [0.45, 0.2, 0.55, 1]
                elif labels[ei, d] == 1:
                    # Feasible — green
                    gt_rgba[row, d] = [0.18, 0.8, 0.44, 1]
                else:
                    # Reachable, tried, failed — gray
                    gt_rgba[row, d] = [0.27, 0.27, 0.27, 1]
            row += 1

    ax.imshow(gt_rgba, aspect="auto", interpolation="nearest")

    for sep in [15, 30, 45]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)
    ax.set_yticks([7, 22, 37, 52])
    ax.set_yticklabels(FACE_NAMES, fontsize=9, fontweight='bold')
    ax.set_xlabel("Push Depth (0-9)")
    ax.set_title("Ground Truth F", fontsize=11, fontweight='bold')

    # Legend for ground truth
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", label="Feasible (opens passage)"),
        mpatches.Patch(facecolor="#444444", label="Tried, didn't clear"),
        mpatches.Patch(facecolor="#73338c", label="Depth blocked (wasted attempt)"),
        mpatches.Patch(facecolor="black", label="Contact point unreachable"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.9)

    # ── Classifier Scores (right) ──
    ax = axes[1]

    # Show scores for all depths at reachable contact points (realistic inference view)
    display = np.full((60, 10), np.nan)
    row = 0
    for edges in FACES_EDGES:
        for ei in edges:
            for d in range(10):
                if cp_reachable[ei, d]:
                    display[row, d] = scores[ei, d]
                # else: stays nan (black)
            row += 1

    masked = np.ma.masked_invalid(display)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Score")

    # Black for unreachable contact points
    unreachable = np.isnan(display)
    black = np.zeros((*display.shape, 4))
    black[unreachable] = [0, 0, 0, 1]
    ax.imshow(black, aspect="auto", interpolation="nearest")

    for sep in [15, 30, 45]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)
    ax.set_yticks([7, 22, 37, 52])
    ax.set_yticklabels(FACE_NAMES, fontsize=9, fontweight='bold')
    ax.set_xlabel("Push Depth (0-9)")
    ax.set_title("Classifier Scores (realistic inference)", fontsize=11, fontweight='bold')

    ratio = result['ratio']
    diff = classify_difficulty(ratio)
    fig.suptitle(f"{title}  |F|/|R| = {result['F']}/{result['R']} = {ratio:.1%}  [{diff.upper().replace('_',' ')}]",
                 fontsize=12, fontweight='bold')

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_face_accuracy(face_acc, output_path):
    """Bar chart of face prediction accuracy by difficulty."""
    difficulties = ["very_hard", "hard", "medium", "easy", "very_easy"]
    diff_labels = ["Very Hard", "Hard", "Medium", "Easy", "Very Easy"]

    accs = []
    counts = []
    for diff in difficulties:
        d = face_acc.get(diff, {'correct': 0, 'total': 0})
        total = d['total']
        acc = d['correct'] / total * 100 if total > 0 else 0
        accs.append(acc)
        counts.append(total)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(diff_labels, accs, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db'],
                  edgecolor='black', linewidth=0.5)

    for bar, acc, n in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.0f}%\n(n={n})', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Face Accuracy (%)')
    ax.set_title('Does the classifier pick the correct object face?', fontweight='bold')
    ax.set_ylim(0, 105)

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate primitive classifier")
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', required=True, help='Path to classifier NPZ files')
    parser.add_argument('--output-dir', default='eval_results', help='Output directory')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-local', action='store_true', default=True)
    parser.add_argument('--n-heatmaps', type=int, default=5,
                        help='Number of heatmap comparisons per difficulty')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = ClassifierModule.load_from_checkpoint(
        args.checkpoint,
        network=DiTClassifier(img_size=args.image_size, in_channels=5),
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

    npz_files = sorted(Path(args.data_dir).rglob('*.npz'))
    print(f"Found {len(npz_files)} NPZ files")

    dataset = ClassifierDataset(
        [str(f) for f in npz_files],
        transform=transform, use_local=args.use_local
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Evaluate
    print("Running evaluation...")
    results = evaluate_model(model, dataloader, device)
    print(f"Evaluated {len(results)} instances")

    # ── Metrics ──
    k_values = [1, 3, 5, 10, 20]
    oracle_acc, realistic_acc, wasted_by_diff = compute_topk_accuracy(results, k_values)
    random_acc = compute_random_baseline(results, k_values)
    face_acc = compute_face_accuracy(results)
    cp_dist = compute_contact_point_distance(results)

    # Print results
    difficulties = ["very_hard", "hard", "medium", "easy", "very_easy"]
    diff_labels = ["Very Hard (<5%)", "Hard (5-15%)", "Medium (15-40%)",
                   "Easy (40-70%)", "Very Easy (>70%)"]

    print("\n" + "=" * 80)
    print("TOP-K ACCURACY (ORACLE): exact per-depth reachability known")
    print("=" * 80)

    for k in k_values:
        print(f"\n--- Top-{k} ---")
        print(f"{'Category':<22} {'Classifier':>12} {'Random':>12} {'Improvement':>12} {'n':>6}")
        print("-" * 66)
        for diff, label in zip(difficulties, diff_labels):
            cls = oracle_acc.get(diff, {}).get(k, [])
            rnd = random_acc.get(diff, {}).get(k, [])
            cls_mean = np.mean(cls) * 100 if cls else 0
            rnd_mean = np.mean(rnd) * 100 if rnd else 0
            imp = cls_mean - rnd_mean
            n = len(cls) if cls else 0
            print(f"{label:<22} {cls_mean:>11.1f}% {rnd_mean:>11.1f}% {imp:>+11.1f}% {n:>6}")

    print("\n" + "=" * 80)
    print("TOP-K ACCURACY (REALISTIC): only contact-point reachability known")
    print("  Depth-unreachable picks count as wasted attempts")
    print("=" * 80)

    for k in k_values:
        print(f"\n--- Top-{k} ---")
        print(f"{'Category':<22} {'Realistic':>12} {'Oracle':>12} {'Random':>12} {'n':>6}")
        print("-" * 66)
        for diff, label in zip(difficulties, diff_labels):
            real = realistic_acc.get(diff, {}).get(k, [])
            orac = oracle_acc.get(diff, {}).get(k, [])
            rnd = random_acc.get(diff, {}).get(k, [])
            real_mean = np.mean(real) * 100 if real else 0
            orac_mean = np.mean(orac) * 100 if orac else 0
            rnd_mean = np.mean(rnd) * 100 if rnd else 0
            n = len(real) if real else 0
            print(f"{label:<22} {real_mean:>11.1f}% {orac_mean:>11.1f}% {rnd_mean:>11.1f}% {n:>6}")

    print("\n" + "=" * 80)
    print("WASTED ATTEMPTS: depth-unreachable picks in top-5")
    print("  (contact point reachable, but depth blocked during push)")
    print("=" * 80)
    for diff, label in zip(difficulties, diff_labels):
        wasted = wasted_by_diff.get(diff, [])
        if wasted:
            print(f"  {label:<22} mean={np.mean(wasted):>4.1f}/5  "
                  f"zero_waste={sum(1 for w in wasted if w==0)/len(wasted)*100:>4.0f}%  (n={len(wasted)})")

    print("\n" + "=" * 80)
    print("FACE ACCURACY")
    print("=" * 80)
    for diff, label in zip(difficulties, diff_labels):
        d = face_acc.get(diff, {'correct': 0, 'total': 0})
        total = d['total']
        acc = d['correct'] / total * 100 if total > 0 else 0
        print(f"  {label:<22} {acc:>5.1f}%  (n={total})")

    print("\n" + "=" * 80)
    print("CONTACT POINT DISTANCE (lower = better, 0 = exact match)")
    print("=" * 80)
    for diff, label in zip(difficulties, diff_labels):
        dists = cp_dist.get(diff, [])
        if dists:
            print(f"  {label:<22} mean={np.mean(dists):>5.1f}  median={np.median(dists):>5.1f}  (n={len(dists)})")

    # ── Plots ──
    print("\nGenerating plots...")
    # Use realistic accuracy for the main plot
    plot_topk_comparison(realistic_acc, random_acc, out / "topk_accuracy_realistic.png")
    plot_topk_comparison(oracle_acc, random_acc, out / "topk_accuracy_oracle.png")
    plot_face_accuracy(face_acc, out / "face_accuracy.png")

    # Heatmap comparisons: sample from each difficulty
    buckets = defaultdict(list)
    for i, r in enumerate(results):
        buckets[classify_difficulty(r['ratio'])].append((i, r))

    for diff in difficulties:
        instances = buckets.get(diff, [])
        if not instances:
            continue
        instances.sort(key=lambda x: x[1]['ratio'])
        n = min(args.n_heatmaps, len(instances))
        indices = np.linspace(0, len(instances) - 1, n, dtype=int)

        heatmap_dir = out / "heatmaps" / diff
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for j, idx in enumerate(indices):
            i, r = instances[idx]
            plot_heatmap_comparison(
                r, heatmap_dir / f"{j+1}_ratio{r['ratio']:.0%}.png",
                title=f"{diff}"
            )

    print(f"\nAll results saved to {out}/")


if __name__ == '__main__':
    main()
