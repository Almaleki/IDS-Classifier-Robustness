#!/usr/bin/env python3
"""Read-only overlap audit for the existing Experiment 1 partition.
Run in PyCharm using the original experiment environment.
Dependencies: numpy, pandas, scipy, scikit-learn, joblib.
No training, balancing, label changes, or split changes are performed.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial import cKDTree

RUN_DIR = Path('/Users/almaleki/Desktop/Research/Research_OCT2025/Codes/PhD_Research_1_CIC2018Dataset/.venv/bin/experiment1_multiple_runs/20260919_164823_076724')
# Leave None to use the exact dataset path recorded during training.
DATASET_OVERRIDE = None
# Descriptive similarity thresholds, not adversarial budgets or traffic equivalence.
NEAR_THRESHOLDS = [0.001, 0.01, 0.05]
QUERY_BATCH_SIZE = 256


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def exact_matches(x, train_idx, test_idx):
    """Hash lookup plus full numerical equality verification; labels excluded."""
    x[x == 0] = 0.0  # Canonicalize signed zero for equality-consistent hashing.
    hashes = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
    th = hashes[train_idx]
    unique, first, counts = np.unique(th, return_index=True, return_counts=True)
    pos = pd.Index(unique).get_indexer(hashes[test_idx])
    match = np.full(len(test_idx), -1, dtype=np.int64)
    possible = np.flatnonzero(pos >= 0)
    for start in range(0, len(possible), 4096):
        q = possible[start:start + 4096]
        candidates = train_idx[first[pos[q]]]
        equal = np.all(x[test_idx[q]] == x[candidates], axis=1)
        match[q[equal]] = candidates[equal]
        # Rare hash collision: check every training record with that hash.
        for k in q[~equal]:
            options = train_idx[th == hashes[test_idx[k]]]
            equal_options = options[np.all(x[options] == x[test_idx[k]], axis=1)]
            if len(equal_options):
                match[k] = equal_options[0]
    return match


def main():
    started = time.time()
    prep_path = RUN_DIR / 'datasets/balanced/preprocessing.joblib'
    split_path = RUN_DIR / 'datasets/balanced/split_indices.npz'
    for p in [prep_path, split_path]:
        if not p.is_file():
            raise FileNotFoundError(f'Missing saved artifact: {p}. Set RUN_DIR to the original run folder.')
    prep = joblib.load(prep_path)
    dataset = Path(DATASET_OVERRIDE) if DATASET_OVERRIDE else Path(prep['dataset_path'])
    if not dataset.is_file():
        raise FileNotFoundError(f'Dataset not found: {dataset}. If moved, set DATASET_OVERRIDE to its new path.')
    print('Checking original dataset SHA-256...', flush=True)
    actual_hash = digest(dataset)
    if actual_hash != prep['dataset_sha256']:
        raise ValueError('Dataset hash differs from training. Use the exact original CSV; do not recreate the split.')
    df = pd.read_csv(dataset, low_memory=False)
    features = list(prep['features'])
    missing = set(features + ['AttackCategory']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    with np.load(split_path, allow_pickle=False) as split:
        train_idx = split['train_indices'].copy()
        test_idx = split['test_indices'].copy()
        source = split['source_row_indices'].copy()
    if len(df) != len(source) or not np.array_equal(source, np.arange(len(df))):
        raise ValueError('Unexpected balanced-dataset row mapping; stop to resolve alignment.')
    joined = np.concatenate([train_idx, test_idx])
    if len(joined) != len(df) or not np.array_equal(np.sort(joined), np.arange(len(df))):
        raise ValueError('Saved train/test indices do not form a disjoint partition of this CSV.')
    if not df['AttackCategory'].isin(prep['classes']).all():
        raise ValueError('Unrecognized or missing class labels.')
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite feature values remain after original preprocessing.')
    out = RUN_DIR / ('overlap_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    labels = df['AttackCategory'].to_numpy()
    # Headers identify candidate metadata only; they do not prove a usable grouping.
    metadata = []
    for col in df.columns:
        key = ''.join(c for c in str(col).lower() if c.isalnum())
        candidate = any(t in key for t in ['timestamp', 'capture', 'session', 'scenario', 'host', 'flowid']) or key in ['srcip', 'dstip', 'sourceip', 'destinationip', 'date', 'day']
        if candidate:
            metadata.append({'column': col, 'dtype': str(df[col].dtype),
                             'nonmissing_rows': int(df[col].notna().sum()),
                             'distinct_nonmissing_values': int(df[col].nunique()),
                             'used_as_model_input': col in features})
    pd.DataFrame(metadata, columns=['column', 'dtype', 'nonmissing_rows', 'distinct_nonmissing_values', 'used_as_model_input']).to_csv(out / 'grouping_metadata_inventory.csv', index=False)
    pd.DataFrame({'feature_order': range(len(features)), 'feature': features}).to_csv(out / 'audited_features.csv', index=False)
    print(f'Checking exact numerical duplicates over all {len(features)} model features...', flush=True)
    exact = exact_matches(x, train_idx, test_idx)
    exact_mask = exact >= 0
    # Exact status and partial results are saved before the potentially longer near search.
    audit = pd.DataFrame({'test_position': np.arange(len(test_idx)), 'test_csv_row_0based': test_idx,
                          'true_class': labels[test_idx], 'exact_train_csv_row_0based': exact,
                          'exact_overlap': exact_mask})
    audit.to_csv(out / 'exact_overlap_by_test_row.csv', index=False)
    print(f'Exact overlap: {exact_mask.sum():,}/{len(test_idx):,} test rows ({100*exact_mask.mean():.2f}%).', flush=True)
    thresholds = sorted(set(float(v) for v in NEAR_THRESHOLDS))
    if not thresholds or thresholds[0] <= 0 or not np.isfinite(thresholds).all():
        raise ValueError('Near thresholds must be positive and finite.')
    max_distance = thresholds[-1]
    z = prep['scaler'].transform(x)  # Float64, original training scaler; no refitting.
    if not np.isfinite(z).all():
        raise ValueError('Non-finite standardized values.')
    del x
    print('Building exact nearest-neighbour search over training features. High-dimensional search may take time.', flush=True)
    tree = cKDTree(np.ascontiguousarray(z[train_idx]), leafsize=40)
    nearest = exact.copy()
    distance = np.full(len(test_idx), np.inf)
    distance[exact_mask] = 0.0
    pending = np.flatnonzero(~exact_mask)
    report_every = max(1, 4096 // QUERY_BATCH_SIZE)
    for batch, start in enumerate(range(0, len(pending), QUERY_BATCH_SIZE)):
        q = pending[start:start + QUERY_BATCH_SIZE]
        d, j = tree.query(z[test_idx[q]], k=1, p=np.inf, eps=0,
                          distance_upper_bound=np.nextafter(max_distance, np.inf), workers=1)
        valid = np.isfinite(d) & (d <= max_distance)
        distance[q[valid]] = d[valid]
        nearest[q[valid]] = train_idx[j[valid]]
        if batch % report_every == 0 or start + QUERY_BATCH_SIZE >= len(pending):
            print(f'Near search: {min(start+QUERY_BATCH_SIZE,len(pending)):,}/{len(pending):,}', flush=True)
    audit['nearest_train_csv_row_0based_within_max_threshold'] = nearest
    audit['nearest_distance_within_max_threshold'] = np.where(np.isfinite(distance), distance, np.nan)
    audit['nearest_reference_class'] = [labels[i] if i >= 0 else '' for i in nearest]
    summary = []
    selections = [('All', np.ones(len(test_idx), dtype=bool))]
    selections += [(c, labels[test_idx] == c) for c in prep['classes']]
    for name, selection in selections:
        n = int(selection.sum())
        for threshold in [0.0] + thresholds:
            mask = exact_mask if threshold == 0 else distance <= threshold
            additional = mask & ~exact_mask
            count = int((selection & mask).sum())
            summary.append({'class': name, 'standardized_linf_threshold': threshold,
                            'test_rows': n, 'overlapping_test_rows': count,
                            'overlapping_test_percent': 100*count/n if n else np.nan,
                            'additional_nonexact_rows': int((selection & additional).sum())})
    audit.to_csv(out / 'overlap_by_test_row.csv', index=False)
    result = pd.DataFrame(summary)
    result.to_csv(out / 'overlap_summary.csv', index=False)
    audit.loc[(nearest >= 0) & (audit.true_class != audit.nearest_reference_class)].to_csv(out / 'different_label_nearest_matches.csv', index=False)
    config = {'dataset_path': str(dataset), 'dataset_sha256': actual_hash, 'run_dir': str(RUN_DIR),
              'feature_count': len(features), 'train_rows': len(train_idx), 'test_rows': len(test_idx),
              'thresholds': thresholds, 'distance': 'L-infinity on float64 features transformed by saved training scaler',
              'search': 'scipy.spatial.cKDTree.query k=1 p=inf eps=0, bounded by maximum threshold',
              'near_counts_include_exact': True, 'labels_used_in_matching': False,
              'metadata_candidate_columns': [r['column'] for r in metadata],
              'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
              'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'seconds': time.time()-started}
    (out / 'audit_configuration.json').write_text(json.dumps(config, indent=2))
    (out / 'README.txt').write_text(
        'Audit of the existing partition only. No training or data modification.\n'
        'Exact duplicates mean equality of all cleaned numerical inputs; labels and non-input metadata are excluded.\n'
        'Near similarity requires every standardized feature to differ by at most the stated threshold.\n'
        'Thresholds 0.001, 0.01 and 0.05 are descriptive choices, not established traffic-equivalence definitions.\n'
        'Near counts include exact overlap; additional_nonexact_rows excludes exact overlap.\n'
        'A missing nearest distance means none was found within the largest threshold, not that no neighbour exists.\n'
        'One nearest reference is retained; ties may have other labels. Different-label export is not an exhaustive label-conflict audit.\n'
        'Shared feature vectors do not prove the same captured event; no overlap does not rule out shared sessions or scenarios.\n'
        'Metadata inventory lists candidate columns only. Grouped evaluation was not performed.\n'
        'Results do not measure the effect of overlap on model accuracy; that would require further controlled evaluation.\n')
    shutil.copy2(Path(__file__), out / 'Exp1_Overlap_Audit.py')
    archive = shutil.make_archive(str(out), 'zip', out)
    print(result[result['class'] == 'All'].to_string(index=False), flush=True)
    print(f'\nCompleted. Share this ZIP:\n{archive}', flush=True)


if __name__ == '__main__':
    main()
