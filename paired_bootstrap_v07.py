#!/usr/bin/env python3
"""Conditional paired bootstrap, by scaffold or by molecule, for QBT v0.7.

No retraining or frozen-test access. By default, resample outer DEVELOPMENT
evaluation scaffold clusters with replacement (preserves molecule correlation
within scaffolds). Alternative --unit molecule resamples positive/negative
molecules separately. Within each seed both models use exactly the same draws.
These intervals are conditional on already fitted models and evaluation split;
not evidence of independent across-seed significance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def die(message):
    raise SystemExit('ERROR: ' + message)


def seed_files(folder):
    found = {}
    for file in sorted(folder.glob('dev_predictions_seed*.csv')):
        match = re.fullmatch(r'dev_predictions_seed(\d+)\.csv', file.name)
        if match:
            seed = int(match.group(1))
            if seed in found:
                die(f'Duplicate seed: {seed}')
            found[seed] = file
    return found


def score(y, p):
    return {
        'pr_auc': float(average_precision_score(y, p)),
        'roc_auc': float(roc_auc_score(y, p)),
    }


def verify_seed(seed, pred, split, dev, controls, models):
    expected = {'inchikey', 'label'} | set(models)
    if not expected.issubset(pred.columns):
        die(f'Seed {seed} missing prediction columns {sorted(expected - set(pred.columns))}')
    if pred.inchikey.isna().any() or pred.inchikey.duplicated().any():
        die(f'Seed {seed} has blank or duplicated eval InChIKeys')
    labels = pd.to_numeric(pred.label, errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        die(f'Seed {seed} has invalid labels')
    if (labels.sum() < 5) or ((1 - labels).sum() < 5):
        die(f'Seed {seed} has too few positives or negatives')
    for name in models:
        x = pd.to_numeric(pred[name], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            die(f'Seed {seed}, model {name}: probabilities not finite in [0,1]')
    req = {'inchikey', 'assignment'}
    if not req.issubset(split.columns):
        die(f'Seed {seed} missing split columns {sorted(req - set(split.columns))}')
    if split.inchikey.duplicated().any():
        die(f'Seed {seed}: duplicated InChIKeys in split file')
    eval_ids = set(split.loc[split.assignment.eq('outer_development_eval'), 'inchikey'])
    if set(pred.inchikey) != eval_ids:
        die(f'Seed {seed}: predictions do not exactly match outer DEVELOPMENT evaluation identifiers')
    dev_seed = dev.loc[dev.seed.eq(seed)].set_index('variant')
    if set(models) - set(dev_seed.index):
        die(f'Seed {seed}: missing rows in development_results.csv')
    if 'n_eval' in dev_seed.columns:
        for model in models:
            if int(dev_seed.at[model, 'n_eval']) != len(pred):
                die(f'Seed {seed}: n_eval mismatch for {model}')
    y = labels.to_numpy(dtype=int)
    values = {model: score(y, pred[model].to_numpy(dtype=float)) for model in models}
    for model in models:
        for metric, column in [('pr_auc', 'pr_auc'), ('roc_auc', 'roc_auc')]:
            v = float(dev_seed.at[model, column])
            if abs(v - values[model][metric]) > 1e-7:
                die(f'Seed {seed}, {model}: saved {metric}={v:.7f} but recomputed {values[model][metric]:.7f}')
    for contrast in controls.itertuples(index=False):
        a, b = contrast.model_a, contrast.model_b
        for m in ['pr_auc', 'roc_auc']:
            got = values[a][m] - values[b][m]
            reference = float(getattr(contrast, 'delta_' + m))
            if abs(got - reference) > 1e-7:
                die(f'Seed {seed}, {contrast.contrast}: saved delta {m}={reference:.7f} vs {got:.7f}')
    eval_scaffolds = split.set_index('inchikey').reindex(pred.inchikey)['scaffold'] if 'scaffold' in split else None
    return y, {model: pred[model].to_numpy(dtype=float) for model in models}, values, eval_scaffolds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='results/gnn_v07_controls_5k', help='Existing v0.7 DEVELOPMENT result folder')
    ap.add_argument('--out', default='results/gnn_v07_bootstrap_5k', help='NEW output directory; never overwritten')
    ap.add_argument('--reps', type=int, default=1000, help='Paired bootstrap replicates PER SEED (default 1000)')
    ap.add_argument('--unit', choices=['scaffold','molecule'], default='scaffold', help='Scaffold-cluster bootstrap (default) or class-stratified molecule bootstrap')
    ap.add_argument('--seed', type=int, default=876210, help='Bootstrap RNG seed')
    args = ap.parse_args()
    if args.reps < 200 or args.seed < 0:
        ap.error('--reps must be >= 200 and --seed >= 0')
    inp, out = Path(args.input), Path(args.out)
    if not inp.is_dir():
        die(f'Input directory not found: {inp}')
    required = ['paired_controls.csv', 'development_results.csv']
    for filename in required:
        if not (inp / filename).is_file():
            die(f'Missing: {inp / filename}')
    if out.exists():
        die(f'Refusing to overwrite existing path: {out}; select a NEW --out')
    controls = pd.read_csv(inp / 'paired_controls.csv')
    dev = pd.read_csv(inp / 'development_results.csv')
    req = {'seed', 'contrast', 'model_a', 'model_b', 'delta_pr_auc', 'delta_roc_auc'}
    if not req.issubset(controls):
        die(f'Invalid controls schema: {sorted(req - set(controls))}')
    pred_files = seed_files(inp)
    if not pred_files:
        die('No dev_predictions_seed*.csv files found')
    if set(pred_files) != set(controls.seed) or set(pred_files) != set(dev.seed):
        die('Seed mismatch across development predictions, controls and results')
    if controls.duplicated(['seed', 'contrast']).any() or dev.duplicated(['seed', 'variant']).any():
        die('Duplicated controls or development-result rows')
    models = sorted(set(controls.model_a) | set(controls.model_b))
    rows = []
    print(f'Paired {args.unit}-level bootstrap; only existing DEVELOPMENT predictions.', flush=True)
    print('No training; frozen v0.3 test is NOT opened.', flush=True)
    for seed, pred_path in pred_files.items():
        pred = pd.read_csv(pred_path)
        split_path = inp / f'dev_split_seed{seed}.csv'
        if not split_path.is_file():
            die(f'Missing split audit: {split_path}')
        split = pd.read_csv(split_path)
        this_controls = controls.loc[controls.seed.eq(seed)].copy()
        y, probabilities, observed, scaffolds = verify_seed(seed, pred, split, dev, this_controls, models)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if args.unit == 'scaffold':
            if scaffolds is None:
                die(f'Seed {seed}: dev_split file missing scaffold column required for cluster bootstrap')
            # Empty Murcko scaffolds (acyclics) form one shared cluster.
            groups = scaffolds.fillna('').astype(str).to_numpy()
            keys = sorted(set(groups))
            cluster_members = [np.flatnonzero(groups == key) for key in keys]
            cluster_n = len(cluster_members)
            if cluster_n < 10:
                die(f'Seed {seed}: only {cluster_n} scaffolds; cannot bootstrap reliably')
        else:
            cluster_n = None
        print(f'Seed {seed}: n={len(y)}, positives={len(pos)}, negatives={len(neg)}, scaffolds={cluster_n}; identity + score checks PASS', flush=True)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, seed]))
        draws = {m: {metric: np.empty(args.reps, dtype=float) for metric in ['pr_auc', 'roc_auc']} for m in models}
        for rep in range(args.reps):
            if args.unit == 'molecule':
                sampled = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                          rng.choice(neg, len(neg), replace=True)])
            else:
                for attempt in range(100):
                    sampled_clusters = rng.integers(0, cluster_n, size=cluster_n)
                    sampled = np.concatenate([cluster_members[i] for i in sampled_clusters])
                    if (y[sampled] == 0).any() and (y[sampled] == 1).any():
                        break
                else:
                    die(f'Seed {seed}: 100 attempts failed to draw both classes')
            yb = y[sampled]
            for model in models:
                result = score(yb, probabilities[model][sampled])
                for metric in ['pr_auc', 'roc_auc']:
                    draws[model][metric][rep] = result[metric]
            if (rep + 1) % 250 == 0 or rep + 1 == args.reps:
                print(f'  {rep+1}/{args.reps} bootstrap draws', flush=True)
        for control in this_controls.itertuples(index=False):
            for metric in ['pr_auc', 'roc_auc']:
                diffs = draws[control.model_a][metric] - draws[control.model_b][metric]
                point = observed[control.model_a][metric] - observed[control.model_b][metric]
                lo, hi = np.quantile(diffs, [0.025, 0.975])
                rows.append({
                    'seed': seed, 'contrast': control.contrast,
                    'model_a': control.model_a, 'model_b': control.model_b,
                    'metric': metric, 'point_delta': point,
                    'ci_2p5': float(lo), 'ci_97p5': float(hi),
                    'interval_excludes_zero': bool(lo > 0 or hi < 0),
                    'bootstrap_fraction_delta_le_zero': float(np.mean(diffs <= 0)),
                    'replicates': args.reps, 'resample_unit': args.unit,
                    'n_eval': len(y), 'n_pos': len(pos), 'n_scaffolds': cluster_n
                })
    out.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'bootstrap_cis.csv', index=False)
    preview = result.loc[(result.metric == 'pr_auc') & (result.contrast.isin([
        'matched DFT minus matched RDKit', 'matched DFT minus random DFT',
        'matched RDKit minus random RDKit', 'matched DFT minus RF']))]
    print('\n=== PR-AUC: 95% WITHIN-SEED PAIRED BOOTSTRAP INTERVALS ===')
    print(preview[['seed','contrast','point_delta','ci_2p5','ci_97p5','interval_excludes_zero']].round(4).to_string(index=False))
    manifest = {
        'source': str(inp), 'evaluation': 'DEVELOPMENT ONLY',
        'method': f'paired nonparametric bootstrap, unit={args.unit}, fixed trained predictions',
        'resampling_unit': args.unit,
        'seed_reps': args.reps, 'rng_seed': args.seed,
        'uncertainty_scope': 'within-seed evaluation sampling conditional on model fit and fixed class counts',
        'important_limitations': ['Overlapping development folds across seeds: no pooled confidence interval',
            'Intervals do not include training or model-selection uncertainty',
            'Molecule resampling fixes class counts; scaffold resampling lets prevalence vary',
            'Scaffold bootstrap groups molecules by Murcko scaffold; empty scaffolds form one cluster',
            'Exploratory comparisons across multiple contrasts; interval excludes zero is not adjusted significance'],
        'input_file_sha256': {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [inp/'paired_controls.csv', inp/'development_results.csv', *pred_files.values()]},
    }
    (out / 'bootstrap_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nSAVED:', out.resolve())


if __name__ == '__main__':
    main()
