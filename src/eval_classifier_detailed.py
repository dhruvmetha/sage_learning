#!/usr/bin/env python3
"""
Detailed classifier evaluation with thresholded predictions.

Thresholds:
  score < 0.45 → predicted NOT success
  score > 0.55 → predicted SUCCESS
  0.45 ≤ score ≤ 0.55 → UNCERTAIN

Computes TP, FP, FN, TN, Precision, Recall, F1 by difficulty.
Generates thresholded heatmaps showing predictions vs ground truth.

Usage:
    python src/eval_classifier_detailed.py \
        --checkpoint /path/to/best.ckpt \
        --data-dir /path/to/classifier_test_npz \
        --output-dir /path/to/eval_detailed
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.data.classifier_data import ClassifierDataset
from src.model.classifier_module import ClassifierModule, PrimitiveClassifierCNN
from src.model.dit.dit_classifier import DiTClassifier
from torchvision import transforms


FACES_EDGES = [
    list(range(0, 30, 2)),
    list(range(1, 30, 2)),
    list(range(30, 60, 2)),
    list(range(31, 60, 2)),
]
FACE_NAMES = ["Top", "Bottom", "Right", "Left"]

THRESH_LOW = 0.45
THRESH_HIGH = 0.55


def classify_difficulty(ratio):
    if ratio < 0.05: return "very_hard"
    elif ratio < 0.15: return "hard"
    elif ratio < 0.40: return "medium"
    elif ratio < 0.70: return "easy"
    else: return "very_easy"


def evaluate_model(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            logits = model(context)
            scores = torch.sigmoid(logits).cpu()
            for i in range(context.shape[0]):
                results.append({
                    'scores': scores[i].numpy(),
                    'labels': batch['f_labels'][i].numpy(),
                    'r_mask': batch['r_mask'][i].numpy(),
                    'cp_reachable': batch['cp_reachable'][i].numpy(),
                    'F': int(batch['F'][i]),
                    'R': int(batch['R'][i]),
                    'ratio': float(batch['ratio'][i]),
                })
    return results


def compute_confusion_matrix(results, use_realistic=True):
    """Compute TP/FP/FN/TN with thresholding, by difficulty.

    Only counts primitives where the classifier makes a definite prediction
    (score < THRESH_LOW or score > THRESH_HIGH). Uncertain predictions are tracked separately.
    """
    by_diff = defaultdict(lambda: {
        'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'uncertain': 0,
        'total_evaluated': 0,
    })

    for r in results:
        mask = r['cp_reachable'] if use_realistic else r['r_mask']
        mask = mask.astype(bool)
        labels = r['labels']
        scores = r['scores']
        diff = classify_difficulty(r['ratio'])

        for ei in range(60):
            for d in range(10):
                if not mask[ei, d]:
                    continue

                by_diff[diff]['total_evaluated'] += 1
                gt = labels[ei, d]
                s = scores[ei, d]

                if s > THRESH_HIGH:
                    pred = 1
                elif s < THRESH_LOW:
                    pred = 0
                else:
                    by_diff[diff]['uncertain'] += 1
                    continue

                if pred == 1 and gt == 1:
                    by_diff[diff]['TP'] += 1
                elif pred == 1 and gt == 0:
                    by_diff[diff]['FP'] += 1
                elif pred == 0 and gt == 1:
                    by_diff[diff]['FN'] += 1
                else:
                    by_diff[diff]['TN'] += 1

    return by_diff


def print_confusion_table(by_diff, title):
    difficulties = ["very_hard", "hard", "medium", "easy", "very_easy"]
    labels = ["Very Hard (<5%)", "Hard (5-15%)", "Medium (15-40%)",
              "Easy (40-70%)", "Very Easy (>70%)"]

    print(f"\n{'='*90}")
    print(title)
    print(f"  Thresholds: score > {THRESH_HIGH} = SUCCESS, score < {THRESH_LOW} = NOT SUCCESS")
    print(f"{'='*90}")
    print(f"{'Category':<22} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7} {'Uncert':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 90)

    total = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'uncertain': 0}

    for diff, label in zip(difficulties, labels):
        d = by_diff.get(diff, {})
        tp = d.get('TP', 0)
        fp = d.get('FP', 0)
        fn = d.get('FN', 0)
        tn = d.get('TN', 0)
        unc = d.get('uncertain', 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{label:<22} {tp:>7} {fp:>7} {fn:>7} {tn:>7} {unc:>7} "
              f"{precision:>6.1%} {recall:>6.1%} {f1:>6.1%}")

        for k in ['TP', 'FP', 'FN', 'TN', 'uncertain']:
            total[k] += d.get(k, 0)

    print("-" * 90)
    tp, fp, fn, tn, unc = total['TP'], total['FP'], total['FN'], total['TN'], total['uncertain']
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{'TOTAL':<22} {tp:>7} {fp:>7} {fn:>7} {tn:>7} {unc:>7} "
          f"{prec:>6.1%} {rec:>6.1%} {f1:>6.1%}")


def plot_thresholded_heatmap(result, output_path, title=""):
    """Three-panel heatmap: ground truth | thresholded prediction | raw scores.

    Thresholded prediction colors:
        Bright green = TP (predicted success, actually success)
        Red          = FP (predicted success, actually failed)
        Orange       = FN (predicted not success, actually success)
        Dark gray    = TN (predicted not success, actually failed)
        Light purple = Uncertain (score between thresholds)
        Dark purple  = Depth blocked (contact point reachable, depth not)
        Black        = Contact point unreachable
    """
    labels = result['labels']
    scores = result['scores']
    r_mask = result['r_mask']
    cp_reachable = result['cp_reachable']

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # ── Panel 1: Ground Truth ──
    ax = axes[0]
    gt_rgba = np.zeros((60, 10, 4))
    row = 0
    for edges in FACES_EDGES:
        for ei in edges:
            for d in range(10):
                if not cp_reachable[ei, d]:
                    gt_rgba[row, d] = [0, 0, 0, 1]
                elif not r_mask[ei, d]:
                    gt_rgba[row, d] = [0.45, 0.2, 0.55, 1]
                elif labels[ei, d] == 1:
                    gt_rgba[row, d] = [0.18, 0.8, 0.44, 1]
                else:
                    gt_rgba[row, d] = [0.27, 0.27, 0.27, 1]
            row += 1
    ax.imshow(gt_rgba, aspect="auto", interpolation="nearest")
    for sep in [15, 30, 45]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)
    ax.set_yticks([7, 22, 37, 52])
    ax.set_yticklabels(FACE_NAMES, fontsize=11, fontweight='bold')
    ax.set_xlabel("Push Depth (0-9)", fontsize=11)
    ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
    legend_gt = [
        mpatches.Patch(facecolor="#2ecc71", label="Feasible (opens passage)"),
        mpatches.Patch(facecolor="#444444", label="Tried, didn't clear"),
        mpatches.Patch(facecolor="#73338c", label="Depth blocked"),
        mpatches.Patch(facecolor="black", label="Contact point unreachable"),
    ]
    ax.legend(handles=legend_gt, loc="upper right", fontsize=9, framealpha=0.95)

    # ── Panel 2: Thresholded Prediction ──
    ax = axes[1]
    pred_rgba = np.zeros((60, 10, 4))
    row = 0
    for edges in FACES_EDGES:
        for ei in edges:
            for d in range(10):
                if not cp_reachable[ei, d]:
                    pred_rgba[row, d] = [0, 0, 0, 1]  # unreachable
                elif not r_mask[ei, d]:
                    # Depth blocked — check if classifier predicted success here (wasted)
                    s = scores[ei, d]
                    if s > THRESH_HIGH:
                        pred_rgba[row, d] = [0.8, 0.2, 0.8, 1]  # magenta = predicted success but depth blocked
                    else:
                        pred_rgba[row, d] = [0.35, 0.15, 0.45, 1]  # dark purple
                else:
                    gt = labels[ei, d]
                    s = scores[ei, d]
                    if THRESH_LOW <= s <= THRESH_HIGH:
                        pred_rgba[row, d] = [0.6, 0.5, 0.7, 1]  # light purple = uncertain
                    elif s > THRESH_HIGH and gt == 1:
                        pred_rgba[row, d] = [0.0, 0.9, 0.3, 1]  # bright green = TP
                    elif s > THRESH_HIGH and gt == 0:
                        pred_rgba[row, d] = [0.9, 0.15, 0.15, 1]  # red = FP
                    elif s < THRESH_LOW and gt == 1:
                        pred_rgba[row, d] = [1.0, 0.6, 0.0, 1]  # orange = FN
                    else:
                        pred_rgba[row, d] = [0.22, 0.22, 0.22, 1]  # dark gray = TN
            row += 1
    ax.imshow(pred_rgba, aspect="auto", interpolation="nearest")
    for sep in [15, 30, 45]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)
    ax.set_yticks([7, 22, 37, 52])
    ax.set_yticklabels(FACE_NAMES, fontsize=11, fontweight='bold')
    ax.set_xlabel("Push Depth (0-9)", fontsize=11)
    ax.set_title(f"Thresholded Prediction (>{THRESH_HIGH}=yes, <{THRESH_LOW}=no)", fontsize=10, fontweight='bold')
    legend_pred = [
        mpatches.Patch(facecolor="#00e64d", label="TP — correct success"),
        mpatches.Patch(facecolor="#e62626", label="FP — false success"),
        mpatches.Patch(facecolor="#ff9900", label="FN — missed success"),
        mpatches.Patch(facecolor="#383838", label="TN — correct reject"),
        mpatches.Patch(facecolor="#997fb3", label="Uncertain (0.45-0.55)"),
        mpatches.Patch(facecolor="#cc33cc", label="Pred success + depth blocked"),
        mpatches.Patch(facecolor="#592973", label="Depth blocked"),
        mpatches.Patch(facecolor="black", label="Unreachable"),
    ]
    ax.legend(handles=legend_pred, loc="upper right", fontsize=8, framealpha=0.95)

    # ── Panel 3: Raw Scores ──
    ax = axes[2]
    display = np.full((60, 10), np.nan)
    row = 0
    for edges in FACES_EDGES:
        for ei in edges:
            for d in range(10):
                if cp_reachable[ei, d]:
                    display[row, d] = scores[ei, d]
            row += 1
    masked = np.ma.masked_invalid(display)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Score")
    unreachable = np.isnan(display)
    black = np.zeros((*display.shape, 4))
    black[unreachable] = [0, 0, 0, 1]
    ax.imshow(black, aspect="auto", interpolation="nearest")
    for sep in [15, 30, 45]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)
    ax.set_yticks([7, 22, 37, 52])
    ax.set_yticklabels(FACE_NAMES, fontsize=11, fontweight='bold')
    ax.set_xlabel("Push Depth (0-9)", fontsize=11)
    ax.set_title("Raw Classifier Scores", fontsize=11, fontweight='bold')

    ratio = result['ratio']
    diff = classify_difficulty(ratio)
    fig.suptitle(f"{title}  |F|/|R| = {result['F']}/{result['R']} = {ratio:.1%}  [{diff.upper().replace('_',' ')}]",
                 fontsize=13, fontweight='bold')

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_precision_recall_bars(by_diff, output_path, title=""):
    difficulties = ["very_hard", "hard", "medium", "easy", "very_easy"]
    diff_labels = ["Very Hard\n(<5%)", "Hard\n(5-15%)", "Medium\n(15-40%)",
                   "Easy\n(40-70%)", "Very Easy\n(>70%)"]

    precs, recs, f1s = [], [], []
    for diff in difficulties:
        d = by_diff.get(diff, {})
        tp, fp, fn = d.get('TP', 0), d.get('FP', 0), d.get('FN', 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precs.append(p * 100)
        recs.append(r * 100)
        f1s.append(f * 100)

    x = np.arange(len(difficulties))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precs, width, label='Precision', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x, recs, width, label='Recall', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, f1s, width, label='F1', color='#e74c3c', edgecolor='black', linewidth=0.5)

    for i, (p, r, f) in enumerate(zip(precs, recs, f1s)):
        ax.text(i - width, p + 1, f'{p:.0f}%', ha='center', va='bottom', fontsize=7)
        ax.text(i, r + 1, f'{r:.0f}%', ha='center', va='bottom', fontsize=7)
        ax.text(i + width, f + 1, f'{f:.0f}%', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels, fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title(f'{title}\nThresholds: >{THRESH_HIGH}=success, <{THRESH_LOW}=not success', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='eval_detailed')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-heatmaps', type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ClassifierModule.load_from_checkpoint(
        args.checkpoint,
        network=DiTClassifier(img_size=args.image_size, in_channels=5),
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

    npz_files = sorted(Path(args.data_dir).rglob('*.npz'))
    print(f"Found {len(npz_files)} NPZ files")

    dataset = ClassifierDataset([str(f) for f in npz_files], transform=transform, use_local=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Running evaluation...")
    results = evaluate_model(model, dataloader, device)
    print(f"Evaluated {len(results)} instances")

    # Confusion matrices
    oracle_cm = compute_confusion_matrix(results, use_realistic=False)
    realistic_cm = compute_confusion_matrix(results, use_realistic=True)

    print_confusion_table(oracle_cm, "CONFUSION MATRIX (ORACLE): exact depth reachability known")
    print_confusion_table(realistic_cm, "CONFUSION MATRIX (REALISTIC): contact-point reachability only")

    # Plots
    print("\nGenerating plots...")
    plot_precision_recall_bars(oracle_cm, out / "precision_recall_oracle.png",
                               title="Precision / Recall / F1 (Oracle)")
    plot_precision_recall_bars(realistic_cm, out / "precision_recall_realistic.png",
                               title="Precision / Recall / F1 (Realistic)")

    # Thresholded heatmaps
    buckets = defaultdict(list)
    for i, r in enumerate(results):
        buckets[classify_difficulty(r['ratio'])].append((i, r))

    for diff in ["very_hard", "hard", "medium", "easy", "very_easy"]:
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
            plot_thresholded_heatmap(
                r, heatmap_dir / f"{j+1}_ratio{r['ratio']:.0%}.png",
                title=diff
            )

    print(f"\nAll results saved to {out}/")


if __name__ == '__main__':
    main()
