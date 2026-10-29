#!/usr/bin/env python3
"""Fast structural tests of the v0.8 protocol; no datasets or GPU required."""
import argparse
import numpy as np
import pandas as pd
from gnn_quantumbridge_v08 import fixed_assignments, paired_deltas, summarize, parse_variants

# Same data/split seed => exact same membership. Run seed is NOT an input.
rows = []
for group in range(90):
    for j in (0, 1):
        rows.append({'inchikey': f'MOCK-{group}-{j}',
                     'scaffold': f'group-{group}',
                     'NR-AhR': int((group*7 + j) % 6 == 0)})
dev = pd.DataFrame(rows)
args = argparse.Namespace(task='NR-AhR', split_seed=31415, min_positives=3)
a = fixed_assignments(dev, args)
b = fixed_assignments(dev.copy(), args)
assert a[4] == b[4] and np.array_equal(a[3], b[3])
assert all(len(a[i]) > 0 for i in range(3))
assert len(set(dev.scaffold.iloc[a[0]]) & set(dev.scaffold.iloc[a[1]])) == 0
assert len(set(dev.scaffold.iloc[a[0]]) & set(dev.scaffold.iloc[a[2]])) == 0
assert len(set(dev.scaffold.iloc[a[1]]) & set(dev.scaffold.iloc[a[2]])) == 0
assert parse_variants('scratch,qmugs_matched,qmugs_matched_rdkit') == [
    'scratch', 'qmugs_matched', 'qmugs_matched_rdkit']

records = []
for seed in [101,202,303]:
    for variant, shift in [('qmugs_matched', .035),
                           ('qmugs_matched_rdkit', .01),
                           ('qmugs_random', .02),
                           ('qmugs_rdkit', .005),
                           ('scratch', 0),
                           ('random_forest', .06)]:
        records.append({'seed':seed,'variant':variant,
                        'pr_auc': .35 + shift, 'roc_auc': .75 + shift,
                        'f1': .3 + shift,'mcc': .2 + shift})
df = pd.DataFrame(records)
paired = paired_deltas(df)
summary, contrasts = summarize(df, paired)
r = contrasts[contrasts.contrast.eq('matched DFT minus matched RDKit')].iloc[0]
assert abs(r.mean_delta_pr_auc - .025) < 1e-10
assert r.positive_runs_pr_auc == 3 and r.runs == 3
assert (summary.runs == 3).all()
print('PASS: fixed scaffold split, disjoint groups, paired init-seed contrasts, primary PR-AUC')
