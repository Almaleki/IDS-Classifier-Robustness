#!/usr/bin/env python3
"""Supplementary Experiment 1: feature-group-disjoint training/evaluation.
Uses the same preselected balanced cohort, architecture, five seeds, optimizer,
epochs, minibatch size and primary attack settings as FullExpV5.py.
No additional balancing or record deletion; group allocation is approximately
80:20, so split class counts need not remain equal. Groups are formed without
labels and allocated using class support only, never model performance.
Zero raw-feature and float32-input cross-partition hash overlap is required.
This is not a capture-day/session split and does not remove all near duplicates.
Original outputs are untouched. A new directory is made for every invocation.
Dependencies: torch pandas numpy scikit-learn scipy matplotlib joblib.
"""
from pathlib import Path
from datetime import datetime
import gc
import hashlib
import json
import platform
import random
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= EDIT DATASET PATHS HERE =================
ORIGINAL_RUN = Path('/Users/almaleki/Desktop/Research/Research_OCT2025/Codes/PhD_Research_1_CIC2018Dataset/.venv/bin/experiment1_multiple_runs/20260919_164823_076724')
BALANCED_DATASET = ORIGINAL_RUN.parents[1] / 'multiclass_balanced.csv'
UNBALANCED_DATASET = Path('/Users/almaleki/Desktop/Research/Research_OCT2025/CSE-CIC-IDS2018/merged_cic2018.csv')
RUN_UNBALANCED = False   # False runs only the three balanced configurations.
OUTPUT_PARENT = ORIGINAL_RUN.parent.parent / 'experiment1_grouped_overlap_control'
# ==========================================================
RUN_SEEDS = [42, 43, 44, 45, 46]
SPLIT_SEED = 42
EPOCHS = 10
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
TRAIN_EPSILON = 0.10
PGD_STEPS = 10
TRAIN_PGD_ALPHA = 0.02
EPSILONS = [0.01, 0.05, 0.10, 0.20, 0.30]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXPECTED_CLASSES = ['Benign', 'Botnet', 'BruteForce', 'DDoS', 'DoS', 'Infiltration']

# Original six-class taxonomy; SQL Injection and unmapped labels are excluded.
LABEL_MAPPING = {
    'Benign': 'Benign', 'FTP-BruteForce': 'BruteForce',
    'SSH-Bruteforce': 'BruteForce', 'Brute Force -Web': 'BruteForce',
    'Brute Force -XSS': 'BruteForce',
    'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Slowloris': 'DoS',
    'DoS attacks-SlowHTTPTest': 'DoS', 'DoS attacks-Hulk': 'DoS',
    'DDoS attacks-LOIC-HTTP': 'DDoS', 'DDOS attack-LOIC-UDP': 'DDoS',
    'DDOS attack-HOIC': 'DDoS', 'SQL Injection': None,
    'Bot': 'Botnet', 'Infilteration': 'Infiltration',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class IDSModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def row_hashes(x):
    x = np.array(x, copy=True)
    x[x == 0] = 0  # Canonical signed zero.
    return pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()


def choose_group_partition(groups, y):
    # Select using class supports only, never model performance.
    _, g = np.unique(groups, return_inverse=True)
    n = int(g.max()) + 1
    if n < 2:
        raise ValueError('Fewer than two distinct feature groups.')
    counts = np.zeros((n, 6), dtype=np.int64)
    np.add.at(counts, (g, y), 1)
    totals = counts.sum(axis=0)
    rng = np.random.default_rng(SPLIT_SEED)
    best = None
    for candidate in range(64):
        selected = rng.permutation(n)[:max(1, int(round(0.20*n))) ]
        tc = counts[selected].sum(axis=0)
        if np.any(tc == 0) or np.any(tc == totals):
            continue
        score = float(np.mean((tc/totals - 0.20)**2))
        if best is None or score < best[0]:
            best = (score, selected.copy(), candidate)
    if best is None:
        raise ValueError('No candidate partition has all six classes in both sets.')
    test_group = np.zeros(n, dtype=bool)
    test_group[best[1]] = True
    test = np.flatnonzero(test_group[g])
    train = np.flatnonzero(~test_group[g])
    return train, test, {'candidate_count': 64, 'selected_candidate_0based': best[2],
                         'support_objective': best[0], 'group_count': n}


def merge_input_collisions(groups, input_hash):
    # Union existing raw-feature groups if float32 scaling creates equal inputs.
    _, g = np.unique(groups, return_inverse=True)
    n = int(g.max()) + 1
    parent = np.arange(n)
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    order = np.argsort(input_hash, kind='stable')
    adjacent = input_hash[order[1:]] == input_hash[order[:-1]]
    left = g[order[:-1]][adjacent]
    right = g[order[1:]][adjacent]
    for a, b in zip(left, right):
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(i) for i in range(n)])
    return roots[g]


def prepare_data(path, name, output_dir):
    from scipy.spatial import cKDTree
    print(f'Loading supplementary cohort: {path}', flush=True)
    saved = joblib.load(ORIGINAL_RUN / 'datasets/balanced/preprocessing.joblib')
    actual_hash = sha256(path)
    if actual_hash != saved['dataset_sha256']:
        raise ValueError('CSV does not match the original training dataset hash.')
    df = pd.read_csv(path, low_memory=False)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['AttackCategory'])
    if list(encoder.classes_) != EXPECTED_CLASSES:
        raise ValueError('Unexpected classes.')
    features = list(saved['features'])
    if len(features) != 79:
        raise ValueError('Expected 79 original features.')
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Invalid cleaned inputs.')
    raw_hash = row_hashes(x)
    groups = raw_hash.copy()
    target = output_dir / 'datasets' / name
    target.mkdir(parents=True)
    history = []
    for iteration in range(10):
        train_idx, test_idx, info = choose_group_partition(groups, y)
        scaler = StandardScaler().fit(x[train_idx])
        z = scaler.transform(x).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError('Non-finite float32 model inputs.')
        input_hash = row_hashes(z)
        overlap = np.isin(input_hash[test_idx], input_hash[train_idx])
        info.update(iteration=iteration, float32_overlapping_test_rows=int(overlap.sum()))
        history.append(info)
        print(f'Grouping pass {iteration+1}: {int(overlap.sum())} overlapping model-input rows', flush=True)
        if not overlap.any():
            break
        groups = merge_input_collisions(groups, input_hash)
    else:
        raise RuntimeError('Grouping did not converge; no training performed.')
    assert not np.intersect1d(raw_hash[train_idx], raw_hash[test_idx]).size
    assert not np.intersect1d(groups[train_idx], groups[test_idx]).size
    assert not np.intersect1d(input_hash[train_idx], input_hash[test_idx]).size
    # Hash collisions conservatively merge extra groups; they cannot split identical rows.
    np.savez_compressed(target/'split_indices.npz', train_indices=train_idx,
                        test_indices=test_idx, source_row_indices=df.index.to_numpy())
    np.savez_compressed(target/'group_ids.npz', group_ids=groups)
    joblib.dump({'scaler': scaler, 'features': features, 'classes': list(encoder.classes_),
                 'split_seed': SPLIT_SEED, 'dataset_sha256': actual_hash,
                 'dataset_path': str(path.resolve()), 'split_method': 'feature-group-disjoint'},
                target/'preprocessing.joblib')
    counts = pd.DataFrame({'class': encoder.classes_, 'total': np.bincount(y,minlength=6),
                          'train': np.bincount(y[train_idx],minlength=6),
                          'test': np.bincount(y[test_idx],minlength=6)})
    counts.to_csv(target/'class_counts.csv',index=False)
    print(counts.to_string(index=False), flush=True)
    print('Checking remaining near overlap before training...', flush=True)
    # Float64 standardized distances, fitted ONLY on the final training partition.
    z64 = scaler.transform(x)
    tree = cKDTree(np.ascontiguousarray(z64[train_idx]), leafsize=40)
    distances = np.full(len(test_idx), np.inf)
    for begin in range(0,len(test_idx),512):
        end = min(begin+512,len(test_idx))
        distances[begin:end], _ = tree.query(z64[test_idx[begin:end]], k=1, p=np.inf,
                                  eps=0, distance_upper_bound=np.nextafter(.05,np.inf), workers=1)
        if begin % 8192 == 0:
            print(f'Near audit: {end:,}/{len(test_idx):,}', flush=True)
    audit = []
    for c in ['All'] + list(encoder.classes_):
        use = np.ones(len(test_idx),dtype=bool) if c == 'All' else y[test_idx] == list(encoder.classes_).index(c)
        for tol in [0.,.001,.01,.05]:
            count = int(np.sum((distances <= tol) & use))
            audit.append({'class': c,'threshold':tol,'test_rows':int(use.sum()),
                          'overlapping_test_rows':count,'percent':100*count/int(use.sum())})
    pd.DataFrame(audit).to_csv(target/'near_overlap_summary.csv',index=False)
    audit_meta = {'raw_exact_overlap_rows':0,'float32_exact_overlap_rows':0,
                  'hash_method':'pandas uint64 row hash; collisions conservatively co-grouped',
                  'grouping_history':history, 'test_fraction':len(test_idx)/len(y),
                  'cohort':'Same previously undersampled balanced CSV; no additional resampling',
                  'limitation':'Not day/session/host-disjoint; near similarities may remain',
                  'scaling':'Refit on final training partition only; raw-space budgets can differ from original split'}
    (target/'split_audit.json').write_text(json.dumps(audit_meta,indent=2))
    print('PASS: zero raw-feature and float32-input overlap. Training begins.',flush=True)
    return (TensorDataset(torch.from_numpy(z[train_idx]),torch.from_numpy(y[train_idx]).long()),
            TensorDataset(torch.from_numpy(z[test_idx]),torch.from_numpy(y[test_idx]).long()),
            list(encoder.classes_),features)


def attack(model, x, y, method, epsilon):
    """Untargeted L-infinity attacks; no feature bounds or random restarts."""
    original = x.detach()
    adv = original.clone()
    iterations = 1 if method == 'FGSM' else PGD_STEPS
    alpha = epsilon if method == 'FGSM' else epsilon / 5
    for _ in range(iterations):
        adv = adv.detach().requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adv), y)
        gradient, = torch.autograd.grad(loss, adv)
        with torch.no_grad():
            adv = adv + alpha * gradient.sign()
            adv = original + (adv - original).clamp(-epsilon, epsilon)
    return adv.detach()


def train(model, train_data, method, run_dir):
    # num_workers=0 and global torch seed control the shuffled minibatch order.
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, observations = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if method != 'clean':
                adv = attack(model, x, y, method, TRAIN_EPSILON)
                x, y = torch.cat([x, adv]), torch.cat([y, y])
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            observations += len(y)
        history.append({'epoch': epoch, 'mean_training_loss': loss_sum / observations})
        print(f'  Epoch {epoch}/{EPOCHS}: loss={history[-1]["mean_training_loss"]:.6f}', flush=True)
    pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)


def evaluate(model, test_data, classes, method, epsilon, run_dir):
    model.eval()
    truth, predictions = [], []
    for x, y in DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        if method != 'clean':
            x = attack(model, x, y, method, epsilon)
        with torch.no_grad():
            predicted = model(x).argmax(dim=1)
        truth.append(y.cpu().numpy())
        predictions.append(predicted.cpu().numpy())
    truth, predictions = np.concatenate(truth), np.concatenate(predictions)
    labels = list(range(len(classes)))
    report = classification_report(truth, predictions, labels=labels,
                                   target_names=classes, output_dict=True, zero_division=0)
    tag = 'clean' if method == 'clean' else f'{method}_eps_{epsilon:.2f}'
    pd.DataFrame(report).T.to_csv(run_dir / f'{tag}_classification_report.csv')
    pd.DataFrame(confusion_matrix(truth, predictions, labels=labels),
                 index=classes, columns=classes).to_csv(run_dir / f'{tag}_confusion_matrix.csv')
    # Rows correspond to saved test_indices, preserving an auditable link to input rows.
    np.savez_compressed(run_dir / f'{tag}_predictions.npz', y_true=truth, y_pred=predictions)
    result = {'attack': method, 'epsilon': epsilon, 'test_support': len(truth),
              'accuracy': accuracy_score(truth, predictions),
              'macro_f1': report['macro avg']['f1-score'],
              'weighted_f1': report['weighted avg']['f1-score'],
              'macro_precision': report['macro avg']['precision'],
              'macro_recall': report['macro avg']['recall']}
    print(f'  {tag}: accuracy={result["accuracy"]:.6f}, macro-F1={result["macro_f1"]:.6f}', flush=True)
    return result


def summarise(rows, output_dir):
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / 'all_run_metrics.csv', index=False)
    keys = ['dataset', 'model', 'attack', 'epsilon']
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
    summary = raw.groupby(keys)[metrics].agg(['count', 'mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(output_dir / 'mean_sd_metrics.csv', index=False)  # pandas std uses ddof=1
    gains = []
    balanced = raw.loc[raw['dataset'] == 'balanced']
    for metric in ['accuracy', 'macro_f1']:
        pivot = balanced.pivot(index=['seed', 'attack', 'epsilon'], columns='model', values=metric)
        if 'baseline' not in pivot.columns:
            continue
        for model in ['FGSM_AT', 'PGD_AT']:
            if model in pivot.columns:
                delta = (100 * (pivot[model] - pivot['baseline'])).dropna().rename('gain_pp').reset_index()
                delta['model'], delta['metric'] = model, metric
                gains.append(delta)
    if gains:
        changes = pd.concat(gains, ignore_index=True)
        changes.to_csv(output_dir / 'paired_gains_by_seed.csv', index=False)
        changes.groupby(['model', 'metric', 'attack', 'epsilon'])['gain_pp'].agg(
            ['count', 'mean', 'std']).reset_index().to_csv(output_dir / 'paired_gains_mean_sd.csv', index=False)
    for method in ['FGSM', 'PGD']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, group in balanced.loc[balanced['attack'] == method].groupby('model'):
            values = group.groupby('epsilon')['accuracy'].agg(['mean', 'std']).sort_index()
            ax.errorbar(values.index, values['mean'], yerr=values['std'].fillna(0),
                        marker='o', capsize=4, label=model)
        ax.set(xlabel='Epsilon (standardised feature space)', ylabel='Accuracy', ylim=(0, 1),
               title=f'{method} robustness: mean ± sample SD across training seeds')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f'{method}_robustness_mean_sd.png', dpi=200)
        plt.close(fig)


def main():
    inputs = [('balanced', BALANCED_DATASET)]
    if RUN_UNBALANCED:
        inputs.append(('unbalanced', UNBALANCED_DATASET))
    for name, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f'{name} dataset not found: {path}. Edit the paths at the top of this script.')
    if len(RUN_SEEDS) < 2 or len(set(RUN_SEEDS)) != len(RUN_SEEDS):
        raise ValueError('Use at least two distinct run seeds.')
    # New output folder every invocation: previous results are never overwritten.
    output_dir = OUTPUT_PARENT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output_dir / 'experiment1_multiple_runs.py')
    metadata = {'seeds': RUN_SEEDS, 'split_seed': SPLIT_SEED, 'epochs': EPOCHS,
                'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
                'train_epsilon': TRAIN_EPSILON, 'train_pgd_alpha': TRAIN_PGD_ALPHA,
                'pgd_steps': PGD_STEPS, 'evaluation_epsilons': EPSILONS,
                'evaluation_pgd_alpha': 'epsilon / 5', 'pgd_random_start': False,
                'pgd_restarts': 0, 'device': str(DEVICE), 'python': sys.version,
                'platform': platform.platform(), 'torch': torch.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'run_unbalanced': RUN_UNBALANCED,
                'scope': 'Supplementary identical-feature-group split on the original balanced cohort; five training seeds; near overlap audited; no session/day separation',
                'group_split_candidates': 64, 'group_target_test_fraction': 0.20}
    (output_dir / 'configuration.json').write_text(json.dumps(metadata, indent=2))
    print(f'Device: {DEVICE}\nOutputs: {output_dir.resolve()}', flush=True)
    rows = []
    reference_features = None
    for dataset_name, path in inputs:
        train_data, test_data, classes, features = prepare_data(path, dataset_name, output_dir)
        if reference_features is None:
            reference_features = features
        elif features != reference_features:
            raise ValueError('Balanced/unbalanced feature names or order differ; check the inputs.')
        configurations = [('baseline', 'clean'), ('FGSM_AT', 'FGSM'), ('PGD_AT', 'PGD')]
        if dataset_name == 'unbalanced':
            configurations = [('baseline_unbalanced', 'clean')]
        for seed in RUN_SEEDS:
            for model_name, method in configurations:
                print(f'\n===== {dataset_name} / {model_name} / seed {seed} =====', flush=True)
                run_dir = output_dir / dataset_name / f'seed_{seed}' / model_name
                run_dir.mkdir(parents=True)
                set_seed(seed)
                model = IDSModel(len(features), len(classes)).to(DEVICE)
                train(model, train_data, method, run_dir)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / 'model.pt')
                evaluations = [('clean', 0.0)]
                if dataset_name == 'balanced':
                    evaluations += [(a, eps) for a in ['FGSM', 'PGD'] for eps in EPSILONS]
                run_rows = []
                for attack_method, epsilon in evaluations:
                    scores = evaluate(model, test_data, classes, attack_method, epsilon, run_dir)
                    scores.update(dataset=dataset_name, model=model_name, seed=seed)
                    run_rows.append(scores)
                rows.extend(run_rows)
                pd.DataFrame(run_rows).to_csv(run_dir / 'metrics.csv', index=False)
                pd.DataFrame(rows).to_csv(output_dir / 'all_run_metrics.csv', index=False)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del train_data, test_data
        gc.collect()
    summarise(rows, output_dir)
    print(f'\nSUCCESS: {len(rows)} evaluation rows saved to {output_dir.resolve()}', flush=True)
    import zipfile
    archive = output_dir / 'review_outputs.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for item in output_dir.rglob('*'):
            if item.is_file() and item != archive and item.suffix != '.pt':
                bundle.write(item,item.relative_to(output_dir))
    print(f'Share this results ZIP: {archive}')


if __name__ == '__main__':
    main()
