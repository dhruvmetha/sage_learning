"""Quick eval: contact point accuracy with raw/R-masked/oracle, for unmasked model."""
import numpy as np, torch, sys
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
from src.data.classifier_data import ClassifierDataset
from src.model.classifier_module import ClassifierModule
from src.model.dit.dit_classifier import DiTClassifier

ckpt = sys.argv[1]
data_dir = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassifierModule.load_from_checkpoint(
    ckpt, network=DiTClassifier(img_size=64, in_channels=5), map_location=device)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Resize((64, 64)),
    transforms.Lambda(lambda x: x * 2 - 1),
])
npz_files = sorted(Path(data_dir).rglob('*.npz'))
dataset = ClassifierDataset([str(f) for f in npz_files], transform=transform, use_local=True)

def classify(r):
    if r < 0.05: return 'very_hard'
    elif r < 0.15: return 'hard'
    elif r < 0.40: return 'medium'
    elif r < 0.70: return 'easy'
    else: return 'very_easy'

by_diff = defaultdict(lambda: {
    'raw_top1': 0, 'raw_top5': 0,
    'r_top1': 0, 'r_top5': 0,
    'oracle_top1': 0, 'oracle_top5': 0,
    'total': 0,
    'first_thresh': [], 'gt_first': [],
})

with torch.no_grad():
    for i in range(len(dataset)):
        sample = dataset[i]
        context = sample['context'].unsqueeze(0).to(device)
        f_labels = sample['f_labels'].numpy()
        r_mask = sample['r_mask'].numpy()
        cp_reachable = sample['cp_reachable'].numpy()
        ratio = sample['ratio']
        diff = classify(ratio)
        if f_labels.sum() == 0: continue
        by_diff[diff]['total'] += 1

        scores = torch.sigmoid(model(context)).squeeze(0).cpu().numpy()
        cp_feasible = np.array([f_labels[ei, :].sum() > 0 for ei in range(60)])

        # Raw
        cp_raw = np.array([scores[ei, :].max() for ei in range(60)])
        ranked = np.argsort(cp_raw)[::-1]
        if cp_feasible[ranked[0]]: by_diff[diff]['raw_top1'] += 1
        if any(cp_feasible[cp] for cp in ranked[:5]): by_diff[diff]['raw_top5'] += 1

        # R masked
        cp_r = np.full(60, -1.0)
        for ei in range(60):
            if cp_reachable[ei, :].sum() > 0: cp_r[ei] = scores[ei, :].max()
        ranked_r = [cp for cp in np.argsort(cp_r)[::-1] if cp_r[cp] > 0]
        if ranked_r and cp_feasible[ranked_r[0]]: by_diff[diff]['r_top1'] += 1
        if any(cp_feasible[cp] for cp in ranked_r[:5]): by_diff[diff]['r_top5'] += 1

        # Oracle
        cp_o = np.full(60, -1.0)
        for ei in range(60):
            rd = r_mask[ei, :].astype(bool)
            if rd.sum() > 0: cp_o[ei] = scores[ei, rd].max()
        ranked_o = [cp for cp in np.argsort(cp_o)[::-1] if cp_o[cp] > 0]
        if ranked_o and cp_feasible[ranked_o[0]]: by_diff[diff]['oracle_top1'] += 1
        if any(cp_feasible[cp] for cp in ranked_o[:5]): by_diff[diff]['oracle_top5'] += 1

        # First depth > 0.55 at top raw cp (if correct)
        top_cp = ranked[0]
        if cp_feasible[top_cp]:
            gt_fd = np.where(f_labels[top_cp, :] == 1)[0]
            pred_fd = np.where(scores[top_cp, :] > 0.55)[0]
            if len(gt_fd) > 0: by_diff[diff]['gt_first'].append(gt_fd.min())
            if len(pred_fd) > 0: by_diff[diff]['first_thresh'].append(pred_fd.min())

print('='*80)
print('CONTACT POINT ACCURACY: Raw vs R-masked vs Oracle')
print('='*80)
print()
header = f'{"":>12} {"n":>5} | {"Raw":>14} | {"R-masked":>14} | {"Oracle":>14}'
print(header)
print(f'{"":>12} {"":>5} | {"top1":>6} {"top5":>6} | {"top1":>6} {"top5":>6} | {"top1":>6} {"top5":>6}')
print('-'*75)
for diff in ['very_hard', 'hard', 'medium', 'easy', 'very_easy']:
    d = by_diff.get(diff)
    if not d or d['total'] == 0: continue
    n = d['total']
    print(f'{diff:>12} {n:>5} | '
          f'{d["raw_top1"]/n*100:>5.0f}% {d["raw_top5"]/n*100:>5.0f}% | '
          f'{d["r_top1"]/n*100:>5.0f}% {d["r_top5"]/n*100:>5.0f}% | '
          f'{d["oracle_top1"]/n*100:>5.0f}% {d["oracle_top5"]/n*100:>5.0f}%')

print()
print('FIRST DEPTH > 0.55 vs GT (at correct raw top-1 contact points):')
for diff in ['very_hard', 'hard', 'medium', 'easy', 'very_easy']:
    d = by_diff.get(diff)
    if not d: continue
    ft = d['first_thresh']
    gt = d['gt_first']
    if ft and gt:
        ft, gt = np.array(ft), np.array(gt)
        exact = (ft == gt).mean() * 100
        print(f'  {diff:>12}: pred={ft.mean():.1f} gt={gt.mean():.1f} exact_match={exact:.0f}% n={len(ft)}')
