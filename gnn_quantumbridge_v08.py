#!/usr/bin/env python3
"""QuantumBridge-Tox v0.8: fixed scaffold split + repeated initialization study.

Development only. Keeps the original v0.3 1,219-molecule test entirely out
of training, threshold selection, model selection and reporting.

One train/inner validation/development evaluation scaffold split is fixed across
all model seeds. QM9 and QMugs training/validation identities are also fixed.
Matched DFT/RDKit share exactly the same pretraining molecules and validation
identities. PR-AUC (average precision) is the prespecified primary metric.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import gnn_quantumbridge as base
import gnn_quantumbridge_v05 as v5
import gnn_quantumbridge_v07 as v7

ALL_VARIANTS = ['scratch', 'qm9_dft', 'qmugs_random', 'qmugs_matched',
                'qmugs_rdkit', 'qmugs_matched_rdkit', 'random_forest']
CONTRASTS = [
    ('qmugs_matched', 'qmugs_matched_rdkit', 'matched DFT minus matched RDKit'),
    ('qmugs_random', 'qmugs_rdkit', 'random DFT minus random RDKit'),
    ('qmugs_matched', 'qmugs_random', 'matched DFT minus random DFT'),
    ('qmugs_matched_rdkit', 'qmugs_rdkit', 'matched RDKit minus random RDKit'),
    ('qmugs_matched', 'scratch', 'matched DFT minus scratch'),
    ('qmugs_matched', 'random_forest', 'matched DFT minus RF'),
]
METRICS = ['pr_auc', 'roc_auc', 'f1', 'mcc']
PRETRAIN_COLS = ['seed', 'variant', 'target', 'n_pretrain', 'n_validation',
                 'best_epoch', 'mae', 'rmse', 'r2']


def parse_variants(text):
    variants = [s.strip() for s in text.split(',') if s.strip()]
    if not variants or len(variants) != len(set(variants)):
        raise argparse.ArgumentTypeError('Use unique comma-separated variant names')
    unexpected = set(variants) - set(ALL_VARIANTS)
    if unexpected:
        raise argparse.ArgumentTypeError(f'Unsupported variant(s): {sorted(unexpected)}')
    return variants


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    p = Path(path).resolve()
    st = p.stat()
    return {'path': str(p), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def fixed_assignments(dev, args):
    """Only one fixed development scaffold split for ALL training seeds."""
    ti, vi, ei = v5.development_split(dev, args, args.split_seed)
    assignment = np.full(len(dev), 'UNASSIGNED', dtype=object)
    assignment[ti] = 'inner_train'
    assignment[vi] = 'inner_earlystop_val'
    assignment[ei] = 'outer_development_eval'
    if (assignment == 'UNASSIGNED').any():
        raise RuntimeError('Incomplete fixed split')
    keys = dev.inchikey.astype(str).to_numpy()
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate Tox21 identities')
    digest = hashlib.sha256('\n'.join(sorted(f'{k}:{a}' for k, a in zip(keys, assignment))).encode()).hexdigest()
    return ti, vi, ei, assignment, digest


def paired_deltas(results):
    rows = []
    for a, b, label in CONTRASTS:
        left = results.loc[results.variant.eq(a)].set_index('seed')
        right = results.loc[results.variant.eq(b)].set_index('seed')
        if len(left) == 0 or len(right) == 0:
            continue
        if set(left.index) != set(right.index):
            raise RuntimeError(f'Unpaired seeds in {label}')
        for seed in sorted(left.index):
            rec = {'seed': int(seed), 'contrast': label, 'model_a': a, 'model_b': b}
            for metric in METRICS:
                rec['delta_' + metric] = float(left.loc[seed, metric] - right.loc[seed, metric])
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(results, paired):
    summary = results.groupby('variant', sort=False).agg(
        runs=('seed', 'count'), mean_pr=('pr_auc', 'mean'), sd_pr=('pr_auc', 'std'),
        mean_roc=('roc_auc', 'mean'), sd_roc=('roc_auc', 'std'),
        mean_f1=('f1', 'mean'), sd_f1=('f1', 'std'),
        mean_mcc=('mcc', 'mean'), sd_mcc=('mcc', 'std')).reset_index()
    if paired.empty:
        ps = pd.DataFrame()
    else:
        records = []
        for name, g in paired.groupby('contrast', sort=False):
            rec = {'contrast': name, 'runs': len(g)}
            for metric in METRICS:
                vals = g['delta_' + metric]
                rec['mean_delta_' + metric] = float(vals.mean())
                rec['sd_delta_' + metric] = float(vals.std()) if len(vals) > 1 else np.nan
                rec['min_delta_' + metric] = float(vals.min())
                rec['max_delta_' + metric] = float(vals.max())
                rec['positive_runs_' + metric] = int((vals > 0).sum())
            records.append(rec)
        ps = pd.DataFrame(records)
    return summary, ps


def atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.part')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def make_manifest(args, heldout_n, partition_hash):
    inputs = {name: file_identity(getattr(args, name))
              for name in ('tox', 'prior_split', 'qmugs_pool')}
    if 'qm9_dft' in args.variants:
        inputs['qm9'] = file_identity(args.qm9)
    inputs['prior_split']['sha256'] = sha256_file(args.prior_split)
    code_dir = Path(__file__).resolve().parent
    code_hashes = {name: sha256_file(code_dir / name) for name in (
        'gnn_quantumbridge.py', 'gnn_quantumbridge_v05.py',
        'gnn_quantumbridge_v07.py', 'gnn_quantumbridge_v08.py')}
    return {'version': 'v0.8', 'status': 'DEVELOPMENT ONLY',
            'code_sha256': code_hashes,
            'primary_metric': 'PR-AUC / average precision, prespecified',
            'test_molecules_excluded': heldout_n,
            'fixed_split_sha256': partition_hash,
            'inputs': inputs,
            'parameters': {
                'seeds': args.seeds, 'split_seed': args.split_seed,
                'selection_seed': args.selection_seed,
                'variants': args.variants,
                'n_pretrain': args.n_pretrain,
                'qm9_val_n': args.qm9_val_n, 'qmugs_val_n': args.qmugs_val_n,
                'qm9_sample_seed': args.qm9_sample_seed,
                'qmugs_sample_seed': args.qmugs_sample_seed,
                'pretrain_epochs': args.pretrain_epochs,
                'finetune_epochs': args.finetune_epochs,
                'patience': args.patience,
                'hidden': args.hidden, 'batch': args.batch,
                'threads': args.threads, 'lr': args.lr,
                'max_atoms': args.max_atoms, 'min_positives': args.min_positives,
                'task': args.task},
            'constraints': [
                'Same fixed train/inner-val/outer-eval scaffold split for every initialization',
                'QMugs selection fixed across seeds; matched DFT and RDKit use same exact molecules',
                'QMugs pretraining validation fixed across seeds and source variants',
                'Frozen original v0.3 test has no fit, selection, evaluation or reporting role',
                'Intervals across init seeds reflect training randomness conditional on one development split only',
                'DFT vs RDKit targets/learning difficulty differ, so result is not a pure causal effect',
                'Only 2D molecular graphs used for conformer-dependent quantum labels',
                'QM9 and QMugs DFT levels and chemical distributions differ',
                'F1/MCC thresholds selected on fixed inner validation, not evaluation fold',
            ]}


def prepare_data(dev, all_tox, args, train_idx):
    pool = pd.read_csv(args.qmugs_pool)
    required = {'smiles', 'inchikey', 'mu', 'homo', 'lumo',
                *v7.MATCH_FEATURES, 'halogen'}
    if not required.issubset(pool.columns):
        raise ValueError(f'QMugs columns missing: {required - set(pool.columns)}')
    if pool.inchikey.isna().any() or pool.inchikey.duplicated().any():
        raise ValueError('QMugs molecular identities must be unique and nonempty')
    tox_keys = set(all_tox.inchikey) | set(
        pd.read_csv(args.tox, usecols=['inchikey']).inchikey.dropna())
    if tox_keys.intersection(pool.inchikey):
        raise ValueError('QMugs pool overlaps Tox21, frozen test may be exposed')
    if len(pool) < args.n_pretrain + args.qmugs_val_n:
        raise ValueError('QMugs pool too small for pretraining and validation')
    val_idx = pool.sample(n=args.qmugs_val_n,
                          random_state=args.qmugs_sample_seed).index
    qmugs_val = pool.loc[val_idx].copy()
    random_set, matched_set = v7.selected_qmugs(
        pool, val_idx, args.selection_seed, dev, train_idx, args.n_pretrain)
    train_hashes = {'qmugs_random': v7.selection_digest(random_set),
                    'qmugs_matched': v7.selection_digest(matched_set)}
    val_hash = v7.selection_digest(qmugs_val)
    if set(qmugs_val.inchikey) & (set(matched_set.inchikey) | set(random_set.inchikey)):
        raise RuntimeError('Pretraining train/validation overlap')
    qm9 = None
    if 'qm9_dft' in args.variants:
        qm9 = v5.prepare_qm9(args, all_tox)
    return qm9, qmugs_val, random_set, matched_set, train_hashes, val_hash


def run_one(seed, args, dev, dev_items, train_ix, val_ix, eval_ix,
            qm9, qmugs_val, random_set, matched_set):
    train = [dev_items[int(i)] for i in train_ix]
    val = [dev_items[int(i)] for i in val_ix]
    eval_items = [dev_items[int(i)] for i in eval_ix]
    predict = pd.DataFrame({
        'inchikey': dev.iloc[eval_ix].inchikey.to_numpy(),
        'label': dev.iloc[eval_ix][args.task].to_numpy()})
    records, pre_records = [], []
    for variant in args.variants:
        v5.set_seed(seed)
        if variant == 'random_forest':
            scores, probs = v5.make_rf(dev, train_ix, val_ix, eval_ix, args, seed)
            info, fit = {}, {}
        else:
            if variant == 'scratch':
                gnn = base.MolecularGNN(hidden=args.hidden, output=1)
                info = {}
            else:
                if variant == 'qm9_dft':
                    qtrain, qval = v7.build_qm9(qm9, args.n_pretrain)
                    targets = v7.DFT3
                else:
                    selected = matched_set if variant in ('qmugs_matched','qmugs_matched_rdkit') else random_set
                    rdkit = variant in ('qmugs_matched_rdkit', 'qmugs_rdkit')
                    targets = ['MolWt', 'TPSA', 'LogP', 'HBD'] if rdkit else v7.DFT3
                    qtrain = v7.build_items(selected, targets, args.max_atoms, rdkit=rdkit)
                    qval = v7.build_items(qmugs_val, targets, args.max_atoms, rdkit=rdkit)
                if len(qtrain) != args.n_pretrain:
                    raise RuntimeError('Pretraining sample count differs')
                encoder = base.MolecularGNN(hidden=args.hidden, output=len(targets))
                print(f'--- run seed={seed}: {variant}, {len(qtrain)} molecules ---', flush=True)
                info = v5.pretrain(encoder, qtrain, qval, targets, args, f'{variant}:{seed}')
                for target in targets:
                    pre_records.append({
                        'seed': seed, 'variant': variant, 'target': target,
                        'n_pretrain': len(qtrain), 'n_validation': len(qval),
                        'best_epoch': info['pretrain_best_epoch'],
                        'mae': info['pretrain_val_mae'][target],
                        'rmse': info['pretrain_val_rmse'][target],
                        'r2': info['pretrain_val_r2'][target]})
                v5.set_seed(seed)
                gnn = base.MolecularGNN(hidden=args.hidden, output=1)
                gnn.embed.load_state_dict(encoder.embed.state_dict())
                gnn.layers.load_state_dict(encoder.layers.state_dict())
                del encoder, qtrain, qval
            v5.set_seed(seed + 917)
            fit = v5.finetune(gnn, train, val, args, f'{variant}:{seed}')
            scores, probs = v5.evaluate(gnn, eval_items, args, fit['threshold'])
            if args.save_checkpoints:
                ck = Path(args.out)/'checkpoints'
                ck.mkdir(exist_ok=True)
                torch.save(gnn.state_dict(), ck/f'{variant}_seed{seed}.pt')
            del gnn
        predict[variant] = probs
        record = {'seed': seed, 'variant': variant, 'n_pretrain':
                  0 if variant in ('scratch', 'random_forest') else args.n_pretrain}
        record.update({k: v for k, v in info.items() if k not in (
            'pretrain_val_mae', 'pretrain_val_rmse', 'pretrain_val_r2')})
        record.update(fit)
        record.update(scores)
        records.append(record)
        print(f'  DONE seed={seed} {variant}: PR-AUC={scores["pr_auc"]:.4f} '
              f'ROC-AUC={scores["roc_auc"]:.4f}', flush=True)
    return pd.DataFrame(records), pd.DataFrame(pre_records, columns=PRETRAIN_COLS), predict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tox', default='data/tox21_clean.csv.gz')
    ap.add_argument('--qm9', default='data/qm9.csv')
    ap.add_argument('--qmugs-pool', default='data/qmugs/qmugs_v06_pool.csv.gz')
    ap.add_argument('--prior-split', default='results/gnn_v03_5k/tox_split.csv')
    ap.add_argument('--task', default='NR-AhR')
    ap.add_argument('--out', default='results/gnn_v08_fixed_split')
    ap.add_argument('--seeds', type=v5.parse_ints, default=[42,123,2026,731,1701])
    ap.add_argument('--split-seed', type=int, default=42)
    ap.add_argument('--selection-seed', type=int, default=42)
    ap.add_argument('--variants', type=parse_variants, default=ALL_VARIANTS[:])
    ap.add_argument('--n-pretrain', type=int, default=5000)
    ap.add_argument('--qm9-val-n', type=int, default=500)
    ap.add_argument('--qmugs-val-n', type=int, default=500)
    ap.add_argument('--qm9-sample-seed', type=int, default=42)
    ap.add_argument('--qmugs-sample-seed', type=int, default=42)
    ap.add_argument('--pretrain-epochs', type=int, default=8)
    ap.add_argument('--finetune-epochs', type=int, default=12)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--max-atoms', type=int, default=128)
    ap.add_argument('--min-positives', type=int, default=12)
    ap.add_argument('--save-checkpoints', action='store_true')
    ap.add_argument('--resume', action='store_true', help='Skip fully saved seeds in a matching run')
    args = ap.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        ap.error('Provide unique nonnegative run seeds')
    if args.split_seed < 0 or args.selection_seed < 0:
        ap.error('Split and selection seeds must be nonnegative')
    if args.n_pretrain < 100 or min(args.qmugs_val_n, args.qm9_val_n) < 20:
        ap.error('Pretraining >= 100 molecules and validation >= 20 required')
    if min(args.pretrain_epochs,args.finetune_epochs,args.batch,args.patience,args.threads) < 1:
        ap.error('Epochs, batch, patience and threads must be positive')
    args.mode = 'multiseed'   # needed by v0.5 prepare_qm9
    args.n_qm9 = args.n_pretrain
    v5.require_files(args)
    if not Path(args.qmugs_pool).is_file():
        ap.error('QMugs pool CSV missing')
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        ap.error(f'Output already exists: {out}. Choose another --out or --resume')
    torch.set_num_threads(args.threads)
    dev, dev_items, all_tox, heldout_n = v5.load_frozen_development(args)
    ti, vi, ei, assignment, partition_hash = fixed_assignments(dev, args)
    manifest = make_manifest(args, heldout_n, partition_hash)
    previous = None
    if args.resume and (out/'manifest.json').is_file():
        previous = json.loads((out/'manifest.json').read_text())
        if any(previous.get(k) != manifest[k] for k in manifest):
            raise RuntimeError('Resume refused: inputs, seeds, split or settings differ')
    elif args.resume and out.exists() and any(out.iterdir()):
        raise RuntimeError('Resume refused: output folder has no matching manifest')
    out.mkdir(parents=True, exist_ok=True)
    split_df = dev[['inchikey','canonical_smiles','scaffold',args.task]].copy()
    split_df['assignment'] = assignment
    if (out/'fixed_dev_split.csv').exists():
        previous = pd.read_csv(out/'fixed_dev_split.csv')
        if not previous.equals(split_df):
            raise RuntimeError('Frozen development split changed during resume')
    else:
        atomic_csv(split_df, out/'fixed_dev_split.csv')
    if previous is None:
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'FIXED DEV SPLIT seed={args.split_seed}: '
          f'train={len(ti)}, inner_val={len(vi)}, outer_dev_eval={len(ei)}; '
          f'positives={int(dev.iloc[ei][args.task].sum())}', flush=True)
    qm9, qval, random_set, matched_set, hashes, val_hash = prepare_data(dev, all_tox, args, ti)
    manifest['qmugs_pretraining_validation_sha256'] = val_hash
    manifest['qmugs_training_selection_sha256'] = hashes
    if previous is not None:
        for k in ('qmugs_pretraining_validation_sha256', 'qmugs_training_selection_sha256'):
            if k in previous and previous[k] != manifest[k]:
                raise RuntimeError(f'Resume refused: {k} differs')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    selections = pd.concat([
        pd.DataFrame({'source': name, 'inchikey': subset.inchikey})
        for name, subset in [('qmugs_random',random_set),('qmugs_matched',matched_set)]
    ], ignore_index=True)
    if (out/'pretraining_selection.csv').exists():
        if not pd.read_csv(out/'pretraining_selection.csv').equals(selections):
            raise RuntimeError('Pretraining identity selection changed during resume')
    else:
        atomic_csv(selections, out/'pretraining_selection.csv')
    selection_audit = pd.DataFrame([
        {'source': name, 'selection_sha256': hashes[name], **v7.audit_selection(subset)}
        for name, subset in [('qmugs_random',random_set),('qmugs_matched',matched_set)]
    ])
    atomic_csv(selection_audit, out/'selection_audit.csv')
    for seed in args.seeds:
        rp, pp, yp = [out/f'{stem}_seed{seed}.csv' for stem in
                      ('results','pretraining_metrics','predictions')]
        if args.resume and rp.exists():
            if not (pp.exists() and yp.exists()):
                raise RuntimeError(f'Seed {seed} has a completed-results marker but missing files')
            old = pd.read_csv(rp)
            pred = pd.read_csv(yp)
            expected = set(args.variants)
            if set(old.variant) == expected and set(pred.columns) == {'inchikey','label',*expected} \
                    and list(pred.inchikey) == list(dev.iloc[ei].inchikey):
                print(f'SKIP completed seed={seed}', flush=True)
                continue
            raise RuntimeError(f'Corrupt saved seed={seed}; inspect before resume')
        if args.resume:
            # An interrupted seed has no results marker (written LAST).
            # Discard only its incomplete script-owned staging files and rerun it.
            for p in (rp, pp, yp):
                for incomplete in (p, p.with_name(p.name + '.part')):
                    if incomplete.exists():
                        print(f'RESTART incomplete seed={seed}, clearing {incomplete.name}', flush=True)
                        incomplete.unlink()
        print(f'\n=== FIXED SPLIT; initialization seed={seed} ===',flush=True)
        r, p, pred = run_one(seed,args,dev,dev_items,ti,vi,ei,qm9,qval,random_set,matched_set)
        atomic_csv(p, pp)
        atomic_csv(pred, yp)
        atomic_csv(r, rp)  # marker of complete seed, written last
    rows = pd.concat([pd.read_csv(out/f'results_seed{s}.csv') for s in args.seeds], ignore_index=True)
    pre = pd.concat([pd.read_csv(out/f'pretraining_metrics_seed{s}.csv')
                     for s in args.seeds],ignore_index=True)
    paired = paired_deltas(rows)
    summary, paired_summary = summarize(rows, paired)
    for name, df in [('development_results.csv', rows),
                     ('pretraining_metrics.csv',pre),('paired_controls.csv',paired),
                     ('paired_controls_summary.csv',paired_summary),('summary.csv',summary)]:
        atomic_csv(df, out/name)
    if sha256_file(args.prior_split) != manifest['inputs']['prior_split']['sha256']:
        raise RuntimeError('Frozen prior split file checksum changed during run')
    print('\n=== FIXED-DEVELOPMENT SUMMARY (NOT FINAL TEST) ===',flush=True)
    print(summary.round(4).to_string(index=False),flush=True)
    print('\n=== PAIRED PR-AUC DELTAS (conditional on fixed development split) ===',flush=True)
    if not paired_summary.empty:
        print(paired_summary[['contrast','runs','mean_delta_pr_auc','sd_delta_pr_auc',
            'positive_runs_pr_auc']].round(4).to_string(index=False),flush=True)
    print('SAVED:',out,flush=True)


if __name__ == '__main__':
    main()
