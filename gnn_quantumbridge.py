#!/usr/bin/env python3
"""CPU-only molecular GNN proof of concept: QM9 DFT pretraining -> Tox21.

The script does not modify or overwrite the project input files.
All output is in results/gnn_v03/ by default.
No torch-geometric or CUDA libraries are required.
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parent
ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
         Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
          Chem.HybridizationType.SP3]
DFT = ['mu', 'alpha', 'homo', 'lumo']  # Nonredundant targets; gap is partly derived


def atom_features(atom):
    z = atom.GetAtomicNum()
    h = atom.GetHybridization()
    return [float(z == x) for x in ELEMENTS] + [float(z not in ELEMENTS),
        float(atom.GetIsAromatic()), float(atom.GetFormalCharge()) / 3,
        float(atom.GetDegree()) / 6, float(atom.GetTotalNumHs()) / 4,
        float(atom.IsInRing())] + [float(h == x) for x in HYBRID] + [
        float(h not in HYBRID), float(atom.GetMass()) / 200.]


def bond_features(bond):
    bt = bond.GetBondType()
    return [float(bt == x) for x in BONDS] + [float(bt not in BONDS),
        float(bond.GetIsConjugated()), float(bond.IsInRing())]

ATOM_DIM = len(atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))
BOND_DIM = 7


def to_graph(smiles, max_atoms=128):
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 1 or mol.GetNumAtoms() > max_atoms:
        return None
    atoms = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src.extend([a, c]); dst.extend([c, a]); ea.extend([f, f])
    edges = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    bonds = torch.tensor(ea, dtype=torch.float32) if ea else torch.empty((0, BOND_DIM), dtype=torch.float32)
    return (atoms, edges, bonds)


def merge_batch(items):
    atom_parts, edge_parts, bond_parts, batch_parts, labels = [], [], [], [], []
    offset = 0
    for i, (graph, label) in enumerate(items):
        x, ei, e = graph
        atom_parts.append(x)
        edge_parts.append(ei + offset)
        bond_parts.append(e)
        batch_parts.append(torch.full((len(x),), i, dtype=torch.long))
        labels.append(torch.as_tensor(label, dtype=torch.float32))
        offset += len(x)
    return (torch.cat(atom_parts), torch.cat(edge_parts, dim=1),
            torch.cat(bond_parts), torch.cat(batch_parts), torch.stack(labels))


class MessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.Linear(hidden + BOND_DIM, hidden)
        self.upd = nn.Linear(hidden * 2, hidden)

    def forward(self, h, edge_index, bond):
        src, dst = edge_index
        m = F.relu(self.msg(torch.cat([h[src], bond], dim=1)))
        agg = torch.zeros_like(h).index_add(0, dst, m)
        deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones((len(dst), 1), dtype=h.dtype, device=h.device))
        return F.relu(self.upd(torch.cat([h, agg / deg.clamp(min=1)], dim=1)))


class MolecularGNN(nn.Module):
    def __init__(self, hidden=32, output=1):
        super().__init__()
        self.embed = nn.Linear(ATOM_DIM, hidden)
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, output))

    def forward(self, atoms, edge_index, bond, graph_ids):
        h = F.relu(self.embed(atoms))
        for layer in self.layers:
            h = h + layer(h, edge_index, bond)
        n = int(graph_ids.max().item()) + 1
        sums = torch.zeros((n, h.shape[1]), dtype=h.dtype, device=h.device).index_add(0, graph_ids, h)
        sizes = torch.bincount(graph_ids, minlength=n).to(h.dtype).unsqueeze(1)
        representations = torch.cat([sums / sizes.clamp(min=1), sums / sizes.clamp(min=1).sqrt()], dim=1)
        return self.head(representations)


def dataset_from_rows(df, label_cols, smiles_col, max_atoms):
    items, accepted = [], []
    for idx, row in df.iterrows():
        graph = to_graph(row[smiles_col], max_atoms=max_atoms)
        if graph is None:
            continue
        y = np.asarray([row[c] for c in label_cols], dtype=np.float32)
        if not np.isfinite(y).all():
            continue
        items.append((graph, torch.tensor(y)))
        accepted.append(idx)
    return df.loc[accepted].reset_index(drop=True), items


def predict(model, items, batch_size=64):
    model.eval()
    arr = []
    with torch.no_grad():
        for batch in DataLoader(items, batch_size=batch_size, collate_fn=merge_batch):
            arr.append(model(*batch[:4]).cpu().numpy())
    return np.concatenate(arr, axis=0) if arr else np.empty((0, 1))


def fit_dft(model, train, val, args):
    labels = np.stack([label.numpy() for _, label in train])
    mean, std = labels.mean(axis=0), labels.std(axis=0).clip(min=1e-6)
    def norm(items):
        return [(g, torch.tensor((y.numpy() - mean) / std, dtype=torch.float32)) for g, y in items]
    tr = norm(train)
    va = norm(val)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')
    best = None
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        losses = []
        for a, ei, b, batch, y in DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=merge_batch):
            optimizer.zero_grad()
            loss = F.mse_loss(model(a, ei, b, batch), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        preds = predict(model, va, args.batch)
        true = np.stack([y.numpy() for _, y in va])
        val_loss = float(np.mean((preds - true) ** 2))
        if val_loss < best_loss:
            best_loss, best = val_loss, copy.deepcopy(model.state_dict())
        print(f'DFT pretrain epoch {epoch}/{args.pretrain_epochs}: train MSE={np.mean(losses):.4f}, val MSE={val_loss:.4f}', flush=True)
    model.load_state_dict(best)
    val_predictions = predict(model, va, args.batch) * std + mean
    val_true = np.stack([y.numpy() for _, y in val])
    maes = np.mean(np.abs(val_predictions - val_true), axis=0)
    return {k: round(float(v), 6) for k, v in zip(DFT, maes)}, mean, std


def fit_toxicity(model, train, val, args, model_name):
    ytrain = np.stack([label.numpy() for _, label in train]).ravel()
    positives, negatives = int(ytrain.sum()), int(len(ytrain)-ytrain.sum())
    if positives < 2 or negatives < 2:
        raise ValueError('Too few positive/negative training samples')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(negatives / positives, 20.), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_true = np.stack([y.numpy() for _, y in val]).ravel().astype(int)
    best_pr, best = -1., None
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        losses = []
        for a, ei, b, batch, y in DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=merge_batch):
            optimizer.zero_grad()
            scores = model(a, ei, b, batch).ravel()
            loss = criterion(scores, y.ravel())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scores = torch.sigmoid(torch.from_numpy(predict(model, val, args.batch).ravel())).numpy()
        val_pr = float(average_precision_score(val_true, scores))
        if val_pr > best_pr:
            best_pr, best = val_pr, copy.deepcopy(model.state_dict())
        print(f'{model_name} epoch {epoch}/{args.finetune_epochs}: train BCE={np.mean(losses):.4f}, val AP={val_pr:.4f}', flush=True)
    model.load_state_dict(best)
    return best_pr


def metrics(y, prob):
    return {'roc_auc': float(roc_auc_score(y, prob)),
            'pr_auc': float(average_precision_score(y, prob)),
            'n': len(y), 'positives': int(np.sum(y)),
            'negative': int(len(y) - np.sum(y))}


def choose_split(y, groups, args, seed, min_positive, min_negative, test_frac):
    splitter = GroupShuffleSplit(n_splits=500, test_size=test_frac, random_state=seed)
    best = None
    for tr, te in splitter.split(np.zeros(len(y)), y, groups):
        ytr, yte = y[tr], y[te]
        if min(ytr.sum(), yte.sum()) < min_positive:
            continue
        if min((1-ytr).sum(), (1-yte).sum()) < min_negative:
            continue
        score = (abs(len(te)/len(y) - test_frac) * 2 +
                 abs(ytr.mean() - yte.mean()) * 2)
        if best is None or score < best[0]:
            best = (score, tr, te)
    if best is None:
        raise ValueError('Could not find a label-balanced scaffold-grouped split; '
                         'try another endpoint or smaller --min-positives.')
    return best[1], best[2]


def load_tox(args):
    tox = pd.read_csv(args.tox)
    needed = {'inchikey', 'canonical_smiles', 'scaffold', args.task}
    if not needed.issubset(tox.columns):
        raise ValueError(f'Missing Tox21 columns: {needed - set(tox.columns)}')
    tox = tox[tox[args.task].isin([0, 1])].dropna(subset=['inchikey','canonical_smiles']).copy()
    # Remove duplicate chemicals with inconsistent toxicity labels, rather than silently retaining first.
    unique_labels = tox.groupby('inchikey')[args.task].nunique(dropna=True)
    conflicts = set(unique_labels[unique_labels > 1].index)
    tox = tox[~tox.inchikey.isin(conflicts)].drop_duplicates('inchikey').reset_index(drop=True)
    tox, data = dataset_from_rows(tox, [args.task], 'canonical_smiles', args.max_atoms)
    if len(tox) < 100:
        raise ValueError('Too few usable toxicity labels')
    print(f'Tox21 task {args.task}: {len(tox)} unique, usable labeled chemicals, '
          f'{int(tox[args.task].sum())} positive; {len(conflicts)} conflicting keys removed', flush=True)
    return tox, data


def load_qm9(args, toxic_keys):
    print('Reading QM9 and generating InChIKeys (may take a minute on CPU)...', flush=True)
    qm = pd.read_csv(args.qm9)
    if not {'smiles', *DFT}.issubset(qm.columns):
        raise ValueError('QM9 is missing expected DFT columns')
    qm = qm.dropna(subset=['smiles'] + DFT).copy()
    for col in DFT:
        qm[col] = pd.to_numeric(qm[col], errors='coerce')
    qm = qm[np.isfinite(qm[DFT].to_numpy()).all(axis=1)].copy()
    keys = []
    for s in qm.smiles:
        try:
            m = Chem.MolFromSmiles(s)
            keys.append(Chem.MolToInchiKey(m) if m is not None else None)
        except Exception:
            keys.append(None)
    qm['inchikey'] = keys
    qm = qm.dropna(subset=['inchikey'])
    # All repeats are removed, not just the 3,636 conflicting subsets.
    duplicated = qm.duplicated('inchikey', keep=False)
    n_ambiguous = int(duplicated.sum())
    qm = qm[~duplicated & ~qm.inchikey.isin(toxic_keys)].copy()
    print(f'QM9 excluded {n_ambiguous} rows with repeated InChIKey and all Tox21 identities; '
          f'{len(qm)} distinct, eligible DFT molecules remain', flush=True)
    if len(qm) < args.n_qm9:
        print(f'WARNING: Only {len(qm)} eligible QM9 records; reducing requested sample', flush=True)
    qm = qm.sample(n=min(args.n_qm9, len(qm)), random_state=args.seed).reset_index(drop=True)
    qm, samples = dataset_from_rows(qm, DFT, 'smiles', args.max_atoms)
    if len(samples) < 100:
        raise ValueError('Too few usable QM9 molecules')
    return qm, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qm9', default='data/qm9.csv')
    ap.add_argument('--tox', default='data/tox21_clean.csv.gz')
    ap.add_argument('--out', default='results/gnn_v03')
    ap.add_argument('--task', default='NR-AhR')
    ap.add_argument('--n-qm9', type=int, default=5000)
    ap.add_argument('--pretrain-epochs', type=int, default=6)
    ap.add_argument('--finetune-epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--max-atoms', type=int, default=128)
    ap.add_argument('--min-positives', type=int, default=12)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.pretrain_epochs < 1 or args.finetune_epochs < 1 or args.batch < 1 or args.n_qm9 < 100:
        ap.error('epochs and batch must be positive; --n-qm9 must be >= 100')
    for path in [args.qm9, args.tox]:
        if not Path(path).is_file():
            ap.error(f'File not found: {path}')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Using CPU; PyTorch {torch.__version__}; workers=0, torch threads={args.threads}', flush=True)

    tox, tox_items = load_tox(args)
    y = tox[args.task].to_numpy(dtype=np.int64)
    groups = tox['scaffold'].fillna(tox['inchikey']).to_numpy(dtype=str)
    trainval_idx, test_idx = choose_split(y, groups, args, args.seed, args.min_positives, 20, .2)
    tr_idx, val_idx_local = choose_split(y[trainval_idx], groups[trainval_idx], args,
                                         args.seed+1, max(5, args.min_positives//3), 10, .15)
    train_idx = trainval_idx[tr_idx]
    val_idx = trainval_idx[val_idx_local]
    train, val, test = ([tox_items[int(i)] for i in ix] for ix in [train_idx, val_idx, test_idx])
    assert not (set(groups[train_idx]) & set(groups[test_idx]))
    assert not (set(groups[val_idx]) & set(groups[test_idx]))
    assert not (set(groups[train_idx]) & set(groups[val_idx]))
    for title, ix in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        print(f'Tox {title}: n={len(ix)}, positive={int(y[ix].sum())}, '
              f'scaffolds={len(set(groups[ix]))}', flush=True)
    # Persist split *before* expensive model training, including exact identities.
    pd.concat([tox.iloc[ix].assign(split=name) for name, ix in
               [('train', train_idx), ('val', val_idx), ('test', test_idx)]])\
       [['inchikey', 'canonical_smiles', 'scaffold', args.task, 'split']]\
       .to_csv(out / 'tox_split.csv', index=False)

    # Strong exclusion: remove every Tox21 identity, including unlabeled rows.
    all_tox_keys = set(pd.read_csv(args.tox, usecols=['inchikey'])['inchikey'].dropna())
    qm, qm_items = load_qm9(args, all_tox_keys)
    qm_train, qm_val = train_test_split(np.arange(len(qm_items)), test_size=.10, random_state=args.seed)
    qtrain = [qm_items[i] for i in qm_train]
    qval = [qm_items[i] for i in qm_val]
    print(f'QM9 pretrain: {len(qtrain)} train + {len(qval)} validation; '
          'NO overlap with any Tox21 identities', flush=True)

    model = MolecularGNN(args.hidden, output=len(DFT))
    dft_mae, means, stds = fit_dft(model, qtrain, qval, args)
    torch.save(model.state_dict(), out / 'dft_pretrained_encoder.pt')
    print(f'QM9 heldout DFT MAE (units as in QM9): {dft_mae}', flush=True)

    # Identical fingerprint baseline using the SAME toxicity split.
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    for s in tox.canonical_smiles:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(s))
        arr = np.zeros((1024,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fps = np.stack(fps)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight='balanced_subsample', n_jobs=args.threads,
                                random_state=args.seed)
    rf.fit(fps[train_idx], y[train_idx])
    rf_prob = rf.predict_proba(fps[test_idx])[:, list(rf.classes_).index(1)]
    records = {'random_forest': metrics(y[test_idx], rf_prob)}
    predictions = pd.DataFrame({'inchikey': tox.iloc[test_idx].inchikey.values,
                                'label': y[test_idx], 'random_forest': rf_prob})
    print('RF on fixed Tox21 test:', records['random_forest'], flush=True)

    for name in ['scratch_gnn', 'dft_pretrained_gnn']:
        torch.manual_seed(args.seed)
        classifier = MolecularGNN(args.hidden, output=1)
        if name == 'dft_pretrained_gnn':
            # Only transfer encoder; DFT regression head is different from classification head.
            classifier.embed.load_state_dict(model.embed.state_dict())
            classifier.layers.load_state_dict(model.layers.state_dict())
        fit_toxicity(classifier, train, val, args, name)
        probs = torch.sigmoid(torch.from_numpy(predict(classifier, test, args.batch).ravel())).numpy()
        predictions[name] = probs
        records[name] = metrics(y[test_idx], probs)
        torch.save(classifier.state_dict(), out / f'{name}.pt')
        print(f'{name} on heldout Tox21 TEST: {records[name]}', flush=True)

    predictions.to_csv(out / 'test_predictions.csv', index=False)
    report = {
        'status': 'pilot, single predefined group-based test split; no statistical proof',
        'task': args.task, 'random_seed': args.seed,
        'tox_total_labeled_unique': int(len(tox)),
        'tox_train_n': len(train), 'tox_val_n': len(val), 'tox_test_n': len(test),
        'qm9_pretrain_n': len(qtrain), 'qm9_validation_n': len(qval),
        'qm9_dft_targets': DFT, 'qm9_validation_dft_mae': dft_mae,
        'qm9_dft_mean_train': dict(zip(DFT, map(float, means))),
        'qm9_dft_std_train': dict(zip(DFT, map(float, stds))),
        'results': records,
        'notes': ['No Tox21 identities in QM9 pretraining, including test identities',
                  'All repeated QM9 InChIKeys are excluded, not first-row retained',
                  'Conflicting Tox21 duplicate labels excluded',
                  'Scaffold grouping uses canonical-structure fallback for acyclic molecules',
                  'Multiple seeds and external validation needed before strong claims']
    }
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n=== FINISHED ===\n' + json.dumps(records, indent=2) +
          f'\nSaved files in: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
