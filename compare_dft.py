#!/usr/bin/env python3
"""Reproduce historical EXPLORATORY NR-AhR comparison.

WARNING: 29 positive labels are clustered in four scaffolds (26 in benzene),
so 5 group shuffles have too few positives in train/test for robust inference.
A positive delta here does not establish that DFT improves toxicity modeling.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
p = ROOT / 'results/qm9_tox21_dft_merged.csv'
df = pd.read_csv(p).dropna(subset=['NR-AhR', 'smiles', 'scaffold']).copy()
dft_cols = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298']
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fp_rows, keep = [], []
for i, smiles in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    arr = np.zeros(1024, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol), arr)
    fp_rows.append(arr)
    keep.append(i)
df = df.iloc[keep].reset_index(drop=True)
X_fp = np.asarray(fp_rows, dtype=np.float32)
X_combo = np.column_stack([X_fp, df[dft_cols].to_numpy(dtype=np.float32)])
y = df['NR-AhR'].astype(int).to_numpy()
groups = df['scaffold'].astype(str).to_numpy()
result = []
for run, (train, test) in enumerate(GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42).split(X_fp, y, groups), 1):
    if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        print(f'Split {run}: one-class train/test; skipped')
        continue
    score = {'split': run, 'train_pos': int(y[train].sum()), 'test_pos': int(y[test].sum())}
    for name, X in [('fingerprint', X_fp), ('fingerprint_dft', X_combo)]:
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced', n_jobs=2, random_state=42)
        m.fit(X[train], y[train])
        prob = m.predict_proba(X[test])[:, 1]
        score[name + '_roc'] = float(roc_auc_score(y[test], prob))
        score[name + '_pr'] = float(average_precision_score(y[test], prob))
    result.append(score)
    print(json.dumps(score))
if not result:
    raise SystemExit('No valid splits')
out = pd.DataFrame(result)
output = ROOT / 'results/dft_comparison.csv'
if output.is_file():
    raise SystemExit('Existing comparison is preserved; rename/move it before rerunning')
out.to_csv(output, index=False)
print('\nMEAN RESULTS (EXPLORATORY ONLY):')
print(out.mean(numeric_only=True).round(4).to_string())
print('DFT ROC delta:', round((out.fingerprint_dft_roc-out.fingerprint_roc).mean(), 4))
print('DFT PR delta:', round((out.fingerprint_dft_pr-out.fingerprint_pr).mean(), 4))
print('CAUTION: positive labels are highly clustered in one scaffold; no robust performance claim.')
