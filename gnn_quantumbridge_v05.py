#!/usr/bin/env python3
"""QuantumBridge-Tox v0.5: development-only molecular model evaluation.

Requires the existing, unchanged gnn_quantumbridge.py (v0.3), the QM9/Tox21
CSVs, and the frozen v0.3 tox_split.csv alongside the user's project.

CRITICAL: v0.3 original test IDs never enter fit, selection, or evaluation.
All numbers produced by this script are *development* estimates, not final tests.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

import gnn_quantumbridge as base  # v0.3, UNMODIFIED

DFT_KEYS = ['mu', 'alpha', 'homo', 'lumo']
VARIANT_TARGETS = {
    'full': ['mu', 'alpha', 'homo', 'lumo'],
    'orbital': ['homo', 'lumo'],
    'response': ['mu', 'alpha'],
    'rdkit': ['MolWt', 'TPSA', 'LogP', 'HBD'],  # generic, NON-DFT control
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def require_files(args):
    for f in (args.tox, args.qm9, args.prior_split):
        if not Path(f).is_file():
            raise FileNotFoundError(f'Required input not found: {f}')
    if not hasattr(base, 'MolecularGNN') or base.DFT != DFT_KEYS:
        raise ValueError('Expected original QuantumBridge-Tox v0.3 gnn_quantumbridge.py')


def safe_output(args):
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output folder already contains files: {out}. Use a NEW --out.')
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_frozen_development(args):
    tox, items = base.load_tox(args)
    prior = pd.read_csv(args.prior_split)
    required = {'inchikey', 'split', args.task}
    if not required.issubset(prior.columns):
        raise ValueError(f'Frozen split missing columns {required - set(prior.columns)}')
    if prior.inchikey.isna().any() or prior.inchikey.duplicated().any():
        raise ValueError('Frozen split must have unique nonempty InChIKeys')
    if not set(prior.split.unique()).issubset({'train', 'val', 'test'}):
        raise ValueError('Unexpected split labels in frozen split')
    if set(tox.inchikey) != set(prior.inchikey):
        raise ValueError('Frozen split differs from the cleaned Tox21 identities; refusing to continue')
    status = prior.set_index('inchikey')['split']
    previous_y = prior.set_index('inchikey')[args.task]
    joined_y = tox['inchikey'].map(previous_y).to_numpy()
    if not np.array_equal(joined_y, tox[args.task].to_numpy()):
        raise ValueError('Toxicity labels differ from those in the frozen v0.3 split')
    tox = tox.assign(prior_split=tox.inchikey.map(status))
    dev_mask = tox.prior_split.isin(['train', 'val']).to_numpy()
    holdout_mask = (tox.prior_split == 'test').to_numpy()
    if not (dev_mask | holdout_mask).all() or not holdout_mask.any():
        raise ValueError('Missing or invalid frozen holdout assignments')
    dev = tox.loc[dev_mask].reset_index(drop=True)
    dev_items = [items[int(i)] for i in np.flatnonzero(dev_mask)]
    heldout_groups = set(tox.loc[holdout_mask, 'scaffold'].fillna(tox.loc[holdout_mask, 'inchikey']))
    dev_groups = set(dev['scaffold'].fillna(dev['inchikey']))
    if heldout_groups & dev_groups:
        raise ValueError('Frozen v0.3 test shares molecular scaffolds with dev pool')
    print(f'Frozen v0.3 test untouched: {int(holdout_mask.sum())} samples. '
          f'Development pool: {len(dev)} samples; positives={int(dev[args.task].sum())}', flush=True)
    return dev, dev_items, tox, int(holdout_mask.sum())


def development_split(dev, args, seed):
    y = dev[args.task].to_numpy(dtype=np.int64)
    groups = dev['scaffold'].fillna(dev['inchikey']).astype(str).to_numpy()
    # Outer development evaluation is separate from early-stopping validation.
    outer_train, outer_eval = base.choose_split(y, groups, args, seed + 10001,
                                                args.min_positives, 10, .20)
    inner_train, inner_val_local = base.choose_split(y[outer_train], groups[outer_train], args,
                                                    seed + 20002, max(3, args.min_positives//3), 8, .20)
    train_idx, val_idx, eval_idx = outer_train[inner_train], outer_train[inner_val_local], outer_eval
    for x, z in [(train_idx, val_idx), (train_idx, eval_idx), (val_idx, eval_idx)]:
        assert not (set(groups[x]) & set(groups[z]))
        assert not (set(dev.inchikey.iloc[x]) & set(dev.inchikey.iloc[z]))
    return train_idx, val_idx, eval_idx


def compute_rdkit_descriptors(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('Unexpected unparseable SMILES in QM9 preprocessing')
    return np.array([Descriptors.MolWt(m), Descriptors.TPSA(m),
                     Descriptors.MolLogP(m), Lipinski.NumHDonors(m)], dtype=np.float32)


def prepare_qm9(args, all_tox):
    # Use original v0.3 deduplication and exhaustive exclusion of ALL Tox21 keys.
    toxic_keys = set(pd.read_csv(args.tox, usecols=['inchikey'])['inchikey'].dropna())
    required_train = max(args.curve_sizes) if args.mode == 'curve' else args.n_qm9
    request_size = required_train + args.qm9_val_n
    loader_args = argparse.Namespace(**vars(args))
    loader_args.n_qm9 = request_size
    loader_args.seed = args.qm9_sample_seed
    qm, qm_items = base.load_qm9(loader_args, toxic_keys)
    if len(qm) < request_size:
        raise ValueError(f'Only {len(qm)} QM9 molecules; requested {request_size}')
    if set(qm.inchikey) & set(all_tox.inchikey):
        raise RuntimeError('QM9 and Tox21 identity overlap detected')
    # Consistent, disjoint 2D-graph sample and fixed QM9 validation set.
    rng = np.random.default_rng(args.qm9_sample_seed + 319)
    perm = rng.permutation(len(qm_items))
    valid_ix, train_ix = perm[:args.qm9_val_n], perm[args.qm9_val_n:]
    # Keep train subset nested for learning curve.
    sample = {'qm': qm, 'items': qm_items, 'train_ix': train_ix, 'val_ix': valid_ix}
    print(f'QM9: {len(train_ix)} available training graphs, {len(valid_ix)} fixed validation graphs; '
          'all disjoint from every Tox21 InChIKey', flush=True)
    return sample


def build_pretrain_items(qm_sample, variant, n):
    keys = VARIANT_TARGETS[variant]
    qm, items = qm_sample['qm'], qm_sample['items']
    needed = np.concatenate([qm_sample['train_ix'][:n], qm_sample['val_ix']])
    if variant == 'rdkit':
        # Computed from structure, not quantum chemistry. This is a CONTROL.
        descriptor_values = {int(i): compute_rdkit_descriptors(qm.iloc[int(i)].smiles) for i in needed}
        get_y = lambda i: descriptor_values[int(i)]
    else:
        vals = qm[keys].to_numpy(dtype=np.float32)
        get_y = lambda i: vals[int(i)]
    def labelled(indices):
        return [(items[int(i)][0], torch.tensor(get_y(i), dtype=torch.float32)) for i in indices]
    return labelled(qm_sample['train_ix'][:n]), labelled(qm_sample['val_ix']), keys


def predict_raw(model, items, batch):
    return base.predict(model, items, batch)


def pretrain(model, train, val, keys, args, name):
    y = np.stack([t.numpy() for _, t in train])
    center, scale = y.mean(axis=0), y.std(axis=0).clip(min=1.e-6)
    train_norm = [(g, torch.tensor((t.numpy() - center)/scale, dtype=torch.float32)) for g,t in train]
    val_norm = [(g, torch.tensor((t.numpy() - center)/scale, dtype=torch.float32)) for g,t in val]
    val_real = np.stack([t.numpy() for _,t in val])
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_loss, best_epoch, stale = None, float('inf'), None, 0
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train(); losses=[]
        for a,ei,b,gid,yy in DataLoader(train_norm, batch_size=args.batch, shuffle=True,
                                        collate_fn=base.merge_batch, num_workers=0):
            optim.zero_grad()
            loss = F.mse_loss(model(a,ei,b,gid), yy)
            loss.backward(); optim.step(); losses.append(float(loss.item()))
        preds = predict_raw(model, val_norm, args.batch)
        true = np.stack([t.numpy() for _,t in val_norm])
        val_loss = float(np.mean((preds - true)**2))
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, best, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        print(f'  {name} pretrain epoch={epoch}: normalized val MSE={val_loss:.4f}', flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    val_pred = predict_raw(model, val_norm, args.batch)*scale + center
    val_mae = {}
    val_rmse = {}
    val_r2 = {}
    for j, key in enumerate(keys):
        val_mae[key] = round(float(mean_absolute_error(val_real[:, j], val_pred[:, j])), 6)
        val_rmse[key] = round(float(np.sqrt(mean_squared_error(val_real[:, j], val_pred[:, j]))), 6)
        if np.isclose(np.var(val_real[:, j]), 0., atol=1e-12):
            val_r2[key] = None  # Undefined for a constant target.
        else:
            val_r2[key] = round(float(r2_score(val_real[:, j], val_pred[:, j])), 6)
    print(f'  {name} best-pretrain metrics MAE={val_mae} RMSE={val_rmse} R2={val_r2}', flush=True)
    return {'pretrain_best_epoch': best_epoch, 'pretrain_val_mse': best_loss,
            'pretrain_val_mae': val_mae, 'pretrain_val_rmse': val_rmse,
            'pretrain_val_r2': val_r2}


def classification_at_threshold(y, prob, threshold):
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if len(y) != len(prob) or len(y) == 0 or not np.isfinite(prob).all():
        raise ValueError('Invalid probability/label arrays')
    pred = prob >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.
    denom = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.
    return {'f1': float(f1), 'mcc': float(mcc),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def choose_validation_threshold(y, probs):
    """Select an MCC-maximizing cutoff using ONLY inner validation labels.

    A predeclared 0.01..0.99 grid avoids searching arbitrary test-driven
    thresholds. Tie break: validation F1 then closeness to 0.5.
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Threshold selection needs both validation classes')
    if not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
        raise ValueError('Invalid validation probabilities')
    candidates = np.linspace(0.01, 0.99, 99)
    scores = [classification_at_threshold(y, probs, t) for t in candidates]
    i = max(range(len(candidates)),
            key=lambda k: (round(scores[k]['mcc'], 12),
                           round(scores[k]['f1'], 12),
                           -abs(float(candidates[k]) - 0.5)))
    return float(round(candidates[i], 4)), scores[i]


def ranking_metrics(y, probs):
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(set(y)) != 2:
        raise ValueError('Evaluation set needs both classes')
    return {'roc_auc': float(roc_auc_score(y, probs)),
            'pr_auc': float(average_precision_score(y, probs)),
            'n_eval': len(y), 'positives_eval': int(y.sum())}


def finetune(model, train, val, args, name):
    y = np.array([float(t.item()) for _,t in train])
    positives = int(y.sum()); negatives = len(y)-positives
    if positives < 2 or negatives < 2:
        raise ValueError('Not enough classes in train')
    val_true = np.array([int(t.item()) for _,t in val])
    if len(np.unique(val_true)) != 2:
        raise ValueError('Not enough classes in early-stopping validation')
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(min(negatives / positives, 20.), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_ap, best_epoch, stale = None, -1., None, 0
    for epoch in range(1, args.finetune_epochs+1):
        model.train()
        for a,ei,b,gid,yy in DataLoader(train, batch_size=args.batch, shuffle=True,
                                        collate_fn=base.merge_batch, num_workers=0):
            optimizer.zero_grad()
            logits = model(a,ei,b,gid).ravel()
            loss = criterion(logits, yy.ravel())
            loss.backward(); optimizer.step()
        val_probs = torch.sigmoid(torch.from_numpy(predict_raw(model,val,args.batch).ravel())).numpy()
        val_ap = float(average_precision_score(val_true, val_probs))
        if val_ap > best_ap + 1e-8:
            best, best_ap, best_epoch, stale = copy.deepcopy(model.state_dict()), val_ap, epoch, 0
        else:
            stale += 1
        print(f'  {name} finetune epoch={epoch}: inner val AP={val_ap:.4f}', flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    selected_probs = torch.sigmoid(
        torch.from_numpy(predict_raw(model, val, args.batch).ravel())).numpy()
    threshold, val_cls = choose_validation_threshold(val_true, selected_probs)
    print(f'  {name} validation-selected MCC threshold={threshold:.2f}; '
          f'inner-val MCC={val_cls["mcc"]:.4f} F1={val_cls["f1"]:.4f}', flush=True)
    return {'finetune_best_epoch': best_epoch, 'inner_val_ap': best_ap,
            'threshold': threshold, 'inner_val_mcc': val_cls['mcc'],
            'inner_val_f1': val_cls['f1'],
            'threshold_selection': 'maximize MCC on inner validation, grid .01-.99'}


def evaluate(model, items, args, threshold):
    probs = torch.sigmoid(torch.from_numpy(predict_raw(model, items, args.batch).ravel())).numpy()
    y = np.array([int(t.item()) for _, t in items])
    values = {**ranking_metrics(y, probs),
              **classification_at_threshold(y, probs, threshold)}
    return values, probs


def make_rf(dev, train_idx, val_idx, eval_idx, args, seed):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    arr=[]
    for smi in dev.canonical_smiles:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smi))
        a = np.zeros(1024, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp,a)
        arr.append(a)
    X = np.stack(arr); y = dev[args.task].to_numpy(dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight='balanced_subsample', n_jobs=args.threads,random_state=seed)
    rf.fit(X[train_idx], y[train_idx])
    positive_col = list(rf.classes_).index(1)
    val_prob = rf.predict_proba(X[val_idx])[:, positive_col]
    threshold, val_cls = choose_validation_threshold(y[val_idx], val_prob)
    prob = rf.predict_proba(X[eval_idx])[:, positive_col]
    result = {**ranking_metrics(y[eval_idx], prob),
              **classification_at_threshold(y[eval_idx], prob, threshold),
              'threshold': threshold, 'inner_val_mcc': val_cls['mcc'],
              'inner_val_f1': val_cls['f1'],
              'threshold_selection': 'maximize MCC on inner validation, grid .01-.99'}
    return result, prob


def configs_for_mode(args):
    if args.mode == 'multiseed':
        return [(v, args.n_qm9) for v in args.variants]
    if args.mode == 'ablation':
        return [(v,args.n_qm9) for v in ['scratch','full','orbital','response','rdkit']]
    if args.mode == 'curve':
        return [('scratch',0)] + [('full',n) for n in args.curve_sizes]
    raise ValueError(args.mode)


def run(args):
    require_files(args)
    out = safe_output(args)
    torch.set_num_threads(args.threads)
    dev, dev_items, all_tox, heldout_n = load_frozen_development(args)
    variants = configs_for_mode(args)
    if 'scratch' not in [v for v,_ in variants]:
        raise ValueError('Include scratch for comparison')
    q = prepare_qm9(args, all_tox)
    prior_sha256 = hashlib.sha256(Path(args.prior_split).read_bytes()).hexdigest()
    report = {'status':'DEVELOPMENT ONLY; original v0.3 test not evaluated',
              'frozen_v03_split_sha256':prior_sha256,
              'input_v03_split':str(Path(args.prior_split).resolve()),
              'input_v03_test_n_excluded':heldout_n,
              'mode': args.mode, 'seeds':args.seeds,
              'qm9_sample_seed':args.qm9_sample_seed,
              'task':args.task, 'model_selection':'inner scaffold-grouped validation AP for epoch; inner validation MCC for threshold',
              'outer_evaluation':'separate development-only scaffold-grouped fold; repeated folds overlap',
              'dft_identity_exclusion':'all Tox21 InChIKeys excluded from QM9',
              'notes':['No final independent performance claim',
                       'Old v0.3 test remains untouched',
                       'Varying random seed changes outer-development partition and model initialization',
                       'RDKit control uses structure-derived descriptors, not DFT labels',
                       'F1/MCC are measured on development evaluation using an INNER-validation MCC-selected cutoff',
                       'Three seeds/overlapping development folds do not establish statistical significance']}
    rows=[]
    pretrain_rows=[]
    for seed in args.seeds:
        train_ix,val_ix,eval_ix = development_split(dev,args,seed)
        assignments = np.full(len(dev), 'UNASSIGNED', dtype=object)
        assignments[train_ix] = 'inner_train'
        assignments[val_ix] = 'inner_earlystop_val'
        assignments[eval_ix] = 'outer_development_eval'
        assert np.all(assignments != 'UNASSIGNED')
        dev[['inchikey','canonical_smiles','scaffold',args.task]].assign(
            assignment=assignments).to_csv(out/f'dev_split_seed{seed}.csv',index=False)
        train=[dev_items[int(i)] for i in train_ix]
        val=[dev_items[int(i)] for i in val_ix]
        evaluation=[dev_items[int(i)] for i in eval_ix]
        print(f'\n=== DEV seed={seed}: train={len(train)}, inner_val={len(val)}, '
              f'outer_dev_eval={len(evaluation)} (positives={sum(int(t.item()) for _,t in evaluation)}) ===',flush=True)
        rf_scores, rf_prob = make_rf(dev,train_ix,val_ix,eval_ix,args,seed)
        rows.append({'seed':seed,'variant':'random_forest','n_qm9':0,
                     'train_n':len(train),'inner_val_n':len(val),**rf_scores})
        predictions = pd.DataFrame({'inchikey':dev.iloc[eval_ix]['inchikey'].to_numpy(),
                                    'label':dev.iloc[eval_ix][args.task].to_numpy(),
                                    'random_forest':rf_prob})
        for variant, n_qm9 in variants:
            set_seed(seed)  # identical initialization for scratch + all GNN variants
            if variant == 'scratch':
                gnn=base.MolecularGNN(hidden=args.hidden,output=1)
                pre_info={}
            else:
                pretrain_train,pretrain_val,targets=build_pretrain_items(q,variant,n_qm9)
                encoder=base.MolecularGNN(hidden=args.hidden,output=len(targets))
                print(f'\n*** seed={seed}, {variant}, QM9 training={n_qm9}; targets={targets} ***', flush=True)
                pre_info=pretrain(encoder,pretrain_train,pretrain_val,targets,args,f'{variant}:{seed}:{n_qm9}')
                for target in targets:
                    pretrain_rows.append({'seed':seed,'variant':variant,'n_qm9':n_qm9,
                                          'target':target,
                                          'mae':pre_info['pretrain_val_mae'][target],
                                          'rmse':pre_info['pretrain_val_rmse'][target],
                                          'r2':pre_info['pretrain_val_r2'][target],
                                          'n_validation':len(pretrain_val),
                                          'best_epoch':pre_info['pretrain_best_epoch']})
                # Keep identical initial toxicity head across variants.
                set_seed(seed)
                gnn=base.MolecularGNN(hidden=args.hidden,output=1)
                gnn.embed.load_state_dict(encoder.embed.state_dict())
                gnn.layers.load_state_dict(encoder.layers.state_dict())
            if variant == 'scratch':
                print(f'\n*** seed={seed}, scratch GNN ***', flush=True)
            set_seed(seed + 917)  # identical shuffled training order among variants, one seed per fold
            fit_info=finetune(gnn,train,val,args,f'{variant}:{seed}:{n_qm9}')
            dev_scores, outer_probs = evaluate(gnn,evaluation,args,fit_info['threshold'])
            predictions[variant + (f'_{n_qm9}' if args.mode == 'curve' else '')] = outer_probs
            rec={'seed':seed,'variant':variant,'n_qm9':n_qm9,
                 'train_n':len(train),'inner_val_n':len(val),**pre_info,**fit_info,**dev_scores}
            rows.append(rec)
            if args.save_checkpoints:
                ck=out/'checkpoints';ck.mkdir(exist_ok=True)
                torch.save(gnn.state_dict(),ck/f'{args.mode}_{variant}_{n_qm9}_seed{seed}.pt')
            print(f'  OUTER DEVELOPMENT metrics: {dev_scores}',flush=True)
        # Durable append per seed, so interrupted runs retain previous completed results.
        predictions.to_csv(out/f'dev_predictions_seed{seed}.csv', index=False)
        pd.DataFrame(rows).to_csv(out/'development_results.csv', index=False)
        if pretrain_rows:
            pd.DataFrame(pretrain_rows).to_csv(out/'pretraining_metrics.csv', index=False)
    df=pd.DataFrame(rows)
    summary=(df.groupby(['variant','n_qm9'],dropna=False)
              .agg(runs=('seed','count'),mean_roc=('roc_auc','mean'),sd_roc=('roc_auc','std'),
                   mean_pr=('pr_auc','mean'),sd_pr=('pr_auc','std'),
                   mean_f1=('f1','mean'),sd_f1=('f1','std'),
                   mean_mcc=('mcc','mean'),sd_mcc=('mcc','std'))
              .reset_index())
    summary.to_csv(out/'summary.csv',index=False)
    scratch = df.loc[df.variant=='scratch',['seed','roc_auc','pr_auc','f1','mcc']].rename(
        columns={'roc_auc':'scratch_roc','pr_auc':'scratch_pr',
                 'f1':'scratch_f1','mcc':'scratch_mcc'})
    paired = df[df.variant!='scratch'].merge(scratch,on='seed',validate='many_to_one')
    paired['delta_roc_vs_scratch'] = paired['roc_auc'] - paired['scratch_roc']
    paired['delta_pr_vs_scratch'] = paired['pr_auc'] - paired['scratch_pr']
    paired['delta_f1_vs_scratch'] = paired['f1'] - paired['scratch_f1']
    paired['delta_mcc_vs_scratch'] = paired['mcc'] - paired['scratch_mcc']
    paired.to_csv(out/'paired_deltas.csv',index=False)
    report.update({'threshold_protocol':'For each model and seed: choose MCC-maximizing threshold from inner validation (0.01–0.99, step 0.01), evaluate F1 and MCC on separate outer development fold; do not tune on outer scores.',
                   'pretrain_metrics_file':'pretraining_metrics.csv',
                   'settings': {'pretrain_epochs':args.pretrain_epochs,
                                'finetune_epochs':args.finetune_epochs,
                                'hidden':args.hidden,'batch':args.batch,
                                'patience':args.patience,'qm9_val_n':args.qm9_val_n},
                   'results':rows})
    (out/'manifest.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x),encoding='utf-8')
    print('\n=== DEVELOPMENT RESULTS (NOT FINAL TEST) ===',flush=True)
    print(summary.round(4).to_string(index=False),flush=True)
    print(f'SAVED: {out}',flush=True)


def parse_ints(s):
    return [int(v.strip()) for v in s.split(',') if v.strip()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tox',default='data/tox21_clean.csv.gz')
    ap.add_argument('--qm9',default='data/qm9.csv')
    ap.add_argument('--prior-split',default='results/gnn_v03_5k/tox_split.csv')
    ap.add_argument('--out',default='results/gnn_v04_dev')
    ap.add_argument('--task',default='NR-AhR')
    ap.add_argument('--mode',choices=['multiseed','ablation','curve'],default='multiseed')
    ap.add_argument('--seeds',type=parse_ints,default=[42,123,2026])
    ap.add_argument('--variants',type=lambda s:[x.strip() for x in s.split(',')],
                    default=['scratch','full','rdkit'])
    ap.add_argument('--curve-sizes',type=parse_ints,default=[1000,5000,10000])
    ap.add_argument('--n-qm9',type=int,default=5000,help='QM9 pretraining samples, excluding validation')
    ap.add_argument('--qm9-val-n',type=int,default=500)
    ap.add_argument('--qm9-sample-seed',type=int,default=42)
    ap.add_argument('--pretrain-epochs',type=int,default=8)
    ap.add_argument('--finetune-epochs',type=int,default=12)
    ap.add_argument('--patience',type=int,default=4)
    ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--max-atoms',type=int,default=128)
    ap.add_argument('--min-positives',type=int,default=12)
    ap.add_argument('--save-checkpoints',action='store_true')
    args=ap.parse_args()
    if not args.seeds or min(args.seeds) < 0:
        ap.error('Provide non-negative seeds')
    if not args.curve_sizes or min(args.curve_sizes) < 100:
        ap.error('Provide curve sizes >=100')
    if args.n_qm9 < 100 or args.qm9_val_n < 20 or args.threads < 1 or args.hidden < 8:
        ap.error('n-qm9 >=100, qm9-val-n >=20, threads>=1, hidden>=8 required')
    if min(args.pretrain_epochs,args.finetune_epochs,args.patience,args.batch) < 1:
        ap.error('Epochs, patience, batch must be positive')
    valid={'scratch',*VARIANT_TARGETS}
    if not set(args.variants).issubset(valid) or len(args.variants)!=len(set(args.variants)):
        ap.error(f'Invalid/duplicate variants; allowed: {sorted(valid)}')
    run(args)


if __name__=='__main__':
    main()
