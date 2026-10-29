#!/usr/bin/env python3
"""QuantumBridge-Tox v0.9: fixed-split, SHUFFLED quantum-target negative control.

Uses EXISTING immutable v0.8 matched-DFT model results and trains ONLY a new
matched-QMugs GNN with randomly reassigned 3-column DFT target ROWS.
Training and validation rows are shuffled separately, with a fixed shuffle seed
across all model-initialization seeds. The real Tox21 NR-AhR labels are NEVER
shuffled. Original v0.3 frozen test remains entirely excluded.

Primary predeclared comparison: real matched DFT minus shuffled matched DFT,
measured by PR-AUC (sklearn average_precision_score) on fixed dev evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, average_precision_score

import gnn_quantumbridge as base
import gnn_quantumbridge_v05 as v5
import gnn_quantumbridge_v07 as v7
import gnn_quantumbridge_v08 as v8

CONTROL = 'qmugs_matched_shuffled_dft'
LABELS = v7.DFT3
BASELINE_NAMES = ['qmugs_matched', 'qmugs_matched_rdkit', 'qmugs_random', 'scratch', 'random_forest']
CONTRASTS = [
    ('qmugs_matched', CONTROL, 'real matched DFT minus shuffled DFT'),
    ('qmugs_matched_rdkit', CONTROL, 'matched RDKit minus shuffled DFT'),
    (CONTROL, 'scratch', 'shuffled DFT minus scratch'),
    ('qmugs_matched', 'qmugs_matched_rdkit', 'real matched DFT minus matched RDKit'),
    ('qmugs_matched', 'qmugs_random', 'real matched DFT minus random DFT'),
    ('qmugs_matched', 'random_forest', 'real matched DFT minus RF'),
]


def shuffle_target_rows(items, seed):
    """Keep molecular graphs fixed; permute WHOLE target vectors jointly.

    This preserves all three label marginal distributions and their covariances.
    It breaks structure-target associations. No global RNG state is modified.
    """
    n = len(items)
    if n < 3:
        raise ValueError('Need at least three molecules to shuffle target rows')
    original = np.stack([y.detach().cpu().numpy() for _, y in items])
    if original.ndim != 2 or original.shape[1] != len(LABELS) or not np.isfinite(original).all():
        raise ValueError('Invalid DFT property vectors')
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    # Extremely unlikely for large n, but ensure negatives truly reassigned.
    if np.array_equal(order, np.arange(n)):
        order = np.roll(order, 1)
    reassigned = [(items[i][0], items[int(j)][1].detach().clone())
                  for i, j in enumerate(order)]
    assert np.allclose(np.stack([t.numpy() for _, t in reassigned]).mean(axis=0), original.mean(axis=0), rtol=1e-5, atol=1e-6)
    assert np.array_equal(np.sort(np.stack([t.numpy() for _, t in reassigned]), axis=0),
                          np.sort(original, axis=0))
    return reassigned, order


def assignment_table(ids, order, partition):
    ids = list(map(str, ids))
    if len(ids) != len(set(ids)) or len(ids) != len(order):
        raise ValueError('Duplicate IDs or length mismatch in shuffle assignment')
    return pd.DataFrame({'partition': partition, 'graph_inchikey': ids,
                         'label_source_inchikey': [ids[int(j)] for j in order]})


def atomic_csv(df, path):
    v8.atomic_csv(df, path)


def validate_existing_v08(source, args):
    manifest_path = source / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing v0.8 manifest: {manifest_path}')
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    if m.get('version') != 'v0.8' or m.get('status') != 'DEVELOPMENT ONLY':
        raise ValueError('Expected a completed DEVELOPMENT-only v0.8 experiment')
    p = m['parameters']
    if args.seeds != p['seeds']:
        raise ValueError(f'Seeds must match existing v0.8 run EXACTLY: {p["seeds"]}')
    if 'qmugs_matched' not in p['variants'] or not all(x in p['variants'] for x in BASELINE_NAMES):
        raise ValueError('Existing v0.8 run needs matched DFT, matched RDKit, random DFT, scratch and RF')
    code_dir = Path(__file__).resolve().parent
    for name, digest in m['code_sha256'].items():
        check = code_dir / name
        if not check.is_file() or v8.sha256_file(check) != digest:
            raise ValueError(f'v0.8 dependency changed after original run: {name}')
    for name, ident in m['inputs'].items():
        path = Path(ident['path'])
        if not path.is_file() or path.stat().st_size != ident['bytes']:
            raise ValueError(f'v0.8 input missing/size changed: {name}: {path}')
        if path.stat().st_mtime_ns != ident['mtime_ns']:
            raise ValueError(f'v0.8 input timestamp changed; audit before continuing: {path}')
        if name == 'prior_split' and v8.sha256_file(path) != ident['sha256']:
            raise ValueError('Frozen prior split checksum changed')
    expected_files = ['fixed_dev_split.csv', 'pretraining_selection.csv', 'selection_audit.csv']
    for seed in args.seeds:
        expected_files += [f'results_seed{seed}.csv', f'predictions_seed{seed}.csv']
    missing = [x for x in expected_files if not (source / x).is_file()]
    if missing:
        raise FileNotFoundError('Existing v0.8 run incomplete: ' + ', '.join(missing))
    return m


def build_args_from_v08(manifest, output):
    p = manifest['parameters']
    i = manifest['inputs']
    d = vars(argparse.Namespace()).copy()
    d.update(p)
    d.update({'tox': i['tox']['path'], 'prior_split': i['prior_split']['path'],
              'qmugs_pool': i['qmugs_pool']['path'],
              'qm9': i.get('qm9', {}).get('path', 'data/qm9.csv'),
              'out': str(output), 'mode': 'multiseed',
              'n_qm9': p['n_pretrain'], 'variants': BASELINE_NAMES})
    return argparse.Namespace(**d)


def evaluate_true_dft_properties(model, real_val, real_train, batch):
    """Diagnostic: predict TRUE (unshuffled) QMugs val properties after null pretrain.

    v5.pretrain trains on NORMALIZED shuffled targets. Permuting rows does not
    change train-column mean and scale, so we can recover physical predictions.
    """
    targets = np.stack([y.numpy() for _, y in real_train])
    center, scale = targets.mean(axis=0), targets.std(axis=0).clip(min=1.e-6)
    preds = base.predict(model, real_val, batch) * scale + center
    truth = np.stack([y.numpy() for _, y in real_val])
    out = []
    for j, name in enumerate(LABELS):
        out.append({'target': name,
                    'mae_on_true_dft': float(mean_absolute_error(truth[:, j], preds[:, j])),
                    'rmse_on_true_dft': float(np.sqrt(mean_squared_error(truth[:, j], preds[:, j]))),
                    'r2_on_true_dft': float(r2_score(truth[:, j], preds[:, j]))})
    return out


def run_seed(seed, settings, dev, dev_items, ti, vi, ei,
             shuffled_train, shuffled_val, real_train, real_val):
    v5.set_seed(seed)
    encoder = base.MolecularGNN(hidden=settings.hidden, output=len(LABELS))
    print(f'--- seed={seed}: shuffled matched DFT, {len(shuffled_train)} train, '
          f'{len(shuffled_val)} val ---', flush=True)
    pretrained = v5.pretrain(encoder, shuffled_train, shuffled_val, LABELS,
                             settings, f'{CONTROL}:{seed}')
    diagnostic = evaluate_true_dft_properties(encoder, real_val, real_train, settings.batch)
    v5.set_seed(seed)
    gnn = base.MolecularGNN(hidden=settings.hidden, output=1)
    gnn.embed.load_state_dict(encoder.embed.state_dict())
    gnn.layers.load_state_dict(encoder.layers.state_dict())
    del encoder
    v5.set_seed(seed + 917)
    train = [dev_items[int(i)] for i in ti]
    val = [dev_items[int(i)] for i in vi]
    eval_items = [dev_items[int(i)] for i in ei]
    fit = v5.finetune(gnn, train, val, settings, f'{CONTROL}:{seed}')
    metrics, probabilities = v5.evaluate(gnn, eval_items, settings, fit['threshold'])
    if settings.save_checkpoints:
        folder = Path(settings.out) / 'checkpoints'
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(gnn.state_dict(), folder / f'{CONTROL}_seed{seed}.pt')
    record = {'seed': seed, 'variant': CONTROL, 'n_pretrain': len(shuffled_train)}
    record.update({k: v for k, v in pretrained.items()
                   if k not in ('pretrain_val_mae', 'pretrain_val_rmse', 'pretrain_val_r2')})
    record.update(fit)
    record.update(metrics)
    pre = []
    for d in diagnostic:
        target = d['target']
        pre.append({'seed': seed, 'variant': CONTROL,
                    'target': target, 'best_epoch': pretrained['pretrain_best_epoch'],
                    'n_pretrain': len(shuffled_train), 'n_validation': len(shuffled_val),
                    'shuffled_validation_mae': pretrained['pretrain_val_mae'][target],
                    'shuffled_validation_rmse': pretrained['pretrain_val_rmse'][target],
                    'shuffled_validation_r2': pretrained['pretrain_val_r2'][target],
                    **d})
    result_pred = pd.DataFrame({'inchikey': dev.iloc[ei].inchikey.to_numpy(),
                                'label': dev.iloc[ei][settings.task].to_numpy(),
                                CONTROL: probabilities})
    return pd.DataFrame([record]), pd.DataFrame(pre), result_pred


def compare_against_v08(source, seeds, shuffled_rows, shuffled_preds):
    model_rows = [shuffled_rows]
    contrasts, pred_combined = [], []
    for seed in seeds:
        base_results = pd.read_csv(source / f'results_seed{seed}.csv')
        base_pred = pd.read_csv(source / f'predictions_seed{seed}.csv')
        new_pred = shuffled_preds[seed]
        if base_pred.inchikey.duplicated().any() or new_pred.inchikey.duplicated().any():
            raise ValueError('Evaluation molecules duplicated')
        if list(base_pred.inchikey) != list(new_pred.inchikey) or not np.array_equal(
                base_pred.label.to_numpy(), new_pred.label.to_numpy()):
            raise ValueError(f'Evaluation sample IDs or labels changed for seed {seed}')
        if len(base_results) != len(base_results.variant.unique()):
            raise ValueError('Duplicated model variants in existing v0.8 results')
        measured = float(average_precision_score(base_pred.label, base_pred.qmugs_matched))
        reported = float(base_results.set_index('variant').at['qmugs_matched', 'pr_auc'])
        if abs(measured-reported) > 1e-8:
            raise ValueError(f'Existing v0.8 matched DFT metric mismatch for seed {seed}')
        src = base_results[base_results.variant.isin(BASELINE_NAMES)].copy()
        src['seed'] = seed
        model_rows.append(src)
        pred = base_pred[['inchikey', 'label'] + BASELINE_NAMES].copy()
        pred[CONTROL] = new_pred[CONTROL].to_numpy()
        pred['seed'] = seed
        pred_combined.append(pred)
        one = pd.concat([src, shuffled_rows[shuffled_rows.seed.eq(seed)]], ignore_index=True)
        idx = one.set_index('variant')
        for a, b, title in CONTRASTS:
            delta = {'seed': seed, 'contrast': title, 'model_a': a, 'model_b': b}
            for metric in v8.METRICS:
                delta['delta_' + metric] = float(idx.at[a, metric] - idx.at[b, metric])
            contrasts.append(delta)
    all_rows = pd.concat(model_rows, ignore_index=True)
    paired = pd.DataFrame(contrasts)
    summary, contrasts_summary = v8.summarize(all_rows, paired)
    return all_rows, pd.concat(pred_combined, ignore_index=True), paired, summary, contrasts_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-v08', default='results/gnn_v08_fixed_split_5k')
    parser.add_argument('--out', default='results/gnn_v09_shuffled_fixed_split_5k')
    parser.add_argument('--shuffle-seed', type=int, default=918273)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-checkpoints', action='store_true')
    args = parser.parse_args()
    if args.shuffle_seed < 0:
        parser.error('shuffle-seed must be nonnegative')
    source = Path(args.from_v08).resolve()
    output = Path(args.out).resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('--out must be an independent directory outside the previous results directory')
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output already exists; use a NEW --out or --resume')
    manifest_old = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    model_seeds = [int(x) for x in manifest_old['parameters']['seeds']]
    tmp_args = argparse.Namespace(seeds=model_seeds)
    manifest_old = validate_existing_v08(source, tmp_args)
    settings = build_args_from_v08(manifest_old, output)
    settings.save_checkpoints = args.save_checkpoints
    v5.require_files(settings)
    torch.set_num_threads(settings.threads)
    dev, dev_items, all_tox, holdout = v5.load_frozen_development(settings)
    ti, vi, ei, assignment, fixed_hash = v8.fixed_assignments(dev, settings)
    if fixed_hash != manifest_old['fixed_split_sha256'] or holdout != manifest_old['test_molecules_excluded']:
        raise ValueError('Fixed development/test split changed since v0.8')
    existing_split = pd.read_csv(source/'fixed_dev_split.csv')
    expected_split = dev[['inchikey','canonical_smiles','scaffold',settings.task]].copy()
    expected_split['assignment'] = assignment
    if not expected_split.equals(existing_split):
        raise ValueError('v0.8 development split CSV differs from reconstructed assignment')
    _, qmugs_val, _, matched, selection_hashes, val_hash = v8.prepare_data(dev, all_tox, settings, ti)
    if selection_hashes != manifest_old['qmugs_training_selection_sha256'] or \
            val_hash != manifest_old['qmugs_pretraining_validation_sha256']:
        raise ValueError('QMugs validation or matched chemistry selection changed since v0.8')
    selected = pd.read_csv(source/'pretraining_selection.csv')
    expected_keys = set(selected.loc[selected.source.eq('qmugs_matched'), 'inchikey'])
    if expected_keys != set(matched.inchikey) or len(expected_keys) != len(matched):
        raise ValueError('Reconstructed matched molecule identities differ from original v0.8')
    true_train = v7.build_items(matched, LABELS, settings.max_atoms)
    true_val = v7.build_items(qmugs_val, LABELS, settings.max_atoms)
    shuffled_train, train_order = shuffle_target_rows(true_train, args.shuffle_seed)
    shuffled_val, val_order = shuffle_target_rows(true_val, args.shuffle_seed + 1)
    assignments = pd.concat([assignment_table(matched.inchikey, train_order, 'train'),
                             assignment_table(qmugs_val.inchikey, val_order, 'validation')], ignore_index=True)
    assign_digest = hashlib.sha256(assignments.to_csv(index=False).encode('utf-8')).hexdigest()
    new_manifest = {
        'version': 'v0.9', 'status': 'DEVELOPMENT ONLY',
        'source_v08_manifest_sha256': v8.sha256_file(source/'manifest.json'),
        'source_v08_path': str(source), 'source_fixed_split_sha256': fixed_hash,
        'shuffle_assignment_sha256': assign_digest, 'shuffle_seed': args.shuffle_seed,
        'model_initialization_seeds': model_seeds,
        'save_checkpoints': args.save_checkpoints,
        'shuffle_scheme': 'independent joint-row permutations in matched QMugs train and val; fixed for all model seeds',
        'true_toxicity_labels_unchanged': True,
        'primary_metric': 'PR-AUC / sklearn average_precision_score',
        'primary_contrast': 'real matched DFT minus shuffled DFT',
        'test_molecules_excluded': holdout,
        'settings': {'pretrain_epochs': settings.pretrain_epochs,
                     'finetune_epochs': settings.finetune_epochs,
                     'patience': settings.patience,
                     'hidden': settings.hidden, 'batch': settings.batch,
                     'lr': settings.lr, 'max_atoms': settings.max_atoms,
                     'n_pretrain': settings.n_pretrain,
                     'qmugs_val_n': settings.qmugs_val_n},
        'code_sha256': v8.sha256_file(__file__),
        'interpretation': ['Conditional on one preselected development split, selection and shuffle mapping',
                           'One shuffled-label permutation is not enough to integrate null permutation variance',
                           'Validation for shuffled control is ALSO shuffled, to avoid selecting on true DFT labels',
                           'Shuffled-label pretraining R2 not comparable to real-target DFT R2',
                           'Original frozen held-out test NOT accessed for evaluation or model choice',
                           'Multiple exploratory contrasts across v0.3-v0.9; no independent statistical claim']
    }
    manifest_file = output/'manifest.json'
    if args.resume:
        if not manifest_file.is_file():
            raise RuntimeError('Cannot resume without matching v0.9 manifest')
        old = json.loads(manifest_file.read_text(encoding='utf-8'))
        for key, value in new_manifest.items():
            if old.get(key) != value:
                raise RuntimeError(f'Resume refused because {key} changed')
        old_assign = pd.read_csv(output/'shuffled_assignments.csv')
        if not old_assign.equals(assignments):
            raise RuntimeError('Resume refused: shuffled target-to-graph assignments changed')
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(new_manifest, indent=2), encoding='utf-8')
        atomic_csv(assignments, output/'shuffled_assignments.csv')
    print(f'v0.9: {holdout} held-out test molecules excluded; fixed dev evaluation {len(ei)} molecules',flush=True)
    print(f'Shuffle fixed across {len(model_seeds)} training seeds. Train changed '
          f'{int(np.sum(train_order != np.arange(len(train_order))))}/{len(train_order)}, '
          f'val changed {int(np.sum(val_order != np.arange(len(val_order))))}/{len(val_order)}',flush=True)
    for seed in model_seeds:
        rp, pp, yp = (output/f'{stem}_seed{seed}.csv' for stem in
                      ('results', 'pretraining_metrics', 'predictions'))
        if args.resume and rp.is_file():
            if not pp.is_file() or not yp.is_file():
                raise RuntimeError(f'Incomplete output marker for seed {seed}')
            prior = pd.read_csv(rp)
            yp_old = pd.read_csv(yp)
            if list(prior.variant) != [CONTROL] or len(yp_old) != len(ei) or \
                    list(yp_old.inchikey) != list(dev.iloc[ei].inchikey):
                raise RuntimeError(f'Existing seed {seed} results invalid')
            print(f'SKIP seed={seed}, existing results complete',flush=True)
            continue
        if args.resume:
            for path in (rp, pp, yp):
                for attempt in (path, path.with_name(path.name + '.part')):
                    if attempt.exists():
                        attempt.unlink()
        print(f'\n=== SHUFFLED NEGATIVE CONTROL seed={seed} ===',flush=True)
        result, pre, pred = run_seed(seed, settings, dev, dev_items, ti, vi, ei,
                                    shuffled_train, shuffled_val, true_train, true_val)
        atomic_csv(pre, pp)
        atomic_csv(pred, yp)
        atomic_csv(result, rp)  # completion marker last
        print(f'  DONE seed={seed} SHUFFLED: PR-AUC={result.iloc[0]["pr_auc"]:.4f}',flush=True)
    new_results = pd.concat([pd.read_csv(output/f'results_seed{seed}.csv') for seed in model_seeds],
                            ignore_index=True)
    new_pre = pd.concat([pd.read_csv(output/f'pretraining_metrics_seed{seed}.csv') for seed in model_seeds],
                        ignore_index=True)
    new_preds = {seed: pd.read_csv(output/f'predictions_seed{seed}.csv') for seed in model_seeds}
    all_results, all_predictions, paired, summary, paired_summary = compare_against_v08(
        source, model_seeds, new_results, new_preds)
    for name, df in [('development_results.csv',all_results),
                     ('pretraining_metrics_shuffled.csv',new_pre),
                     ('predictions_all_seeds.csv',all_predictions),
                     ('paired_controls.csv',paired),
                     ('summary.csv',summary),
                     ('paired_controls_summary.csv',paired_summary)]:
        atomic_csv(df, output/name)
    if v8.sha256_file(source/'manifest.json') != new_manifest['source_v08_manifest_sha256']:
        raise RuntimeError('v0.8 reference manifest changed during run')
    print('\n=== DEV-ONLY METRIC SUMMARY ===',flush=True)
    print(summary[['variant','runs','mean_pr','sd_pr','mean_roc']].round(4).to_string(index=False),flush=True)
    print('\n=== FIXED-SPLIT PAIRED PR-AUC DELTAS ===',flush=True)
    print(paired_summary[['contrast','runs','mean_delta_pr_auc','sd_delta_pr_auc',
                          'positive_runs_pr_auc']].round(4).to_string(index=False),flush=True)
    print('SAVED:',output,flush=True)


if __name__ == '__main__':
    main()
