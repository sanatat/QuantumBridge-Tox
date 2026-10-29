#!/usr/bin/env python3
"""Read existing v0.8/v0.9 DEVELOPMENT CSVs. Do not retrain or touch held-out test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHUFFLES = {
    '918273': 'gnn_v09_shuffled_fixed_split_5k',
    '271828': 'gnn_v09_shuffle_271828',
    '314159': 'gnn_v09_shuffle_314159',
}

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='results/research_report_audit')
    a = ap.parse_args()
    base = Path(a.results)
    v08 = base / 'gnn_v08_fixed_split_5k'
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'Output folder already exists: {out}. Select a fresh --out.')
    manifest = json.loads((v08 / 'manifest.json').read_text())
    if manifest.get('version') != 'v0.8' or manifest.get('status') != 'DEVELOPMENT ONLY':
        raise RuntimeError('Unexpected reference manifest; refuse to mix runs')
    raw = pd.read_csv(v08 / 'development_results.csv')
    if raw.duplicated(['seed','variant']).any():
        raise RuntimeError('Duplicated (seed, variant)')
    runs = sorted(raw.seed.unique().tolist())
    if len(runs) != 5:
        raise RuntimeError('Expected five training seeds')
    if len(set(manifest['parameters']['seeds']) - set(runs)):
        raise RuntimeError('Manifest seeds are not present')
    paired = pd.read_csv(v08 / 'paired_controls.csv')
    fixed = pd.read_csv(v08 / 'fixed_dev_split.csv')
    if fixed.inchikey.duplicated().any():
        raise RuntimeError('Fixed development split has duplicate keys')
    counts = fixed.groupby('assignment').size().to_dict()
    if counts.get('outer_development_eval') != 942:
        raise RuntimeError('Expected 942 v0.8 development-evaluation cases')
    sel = pd.read_csv(v08 / 'pretraining_selection.csv')
    rows = []
    comparisons = []
    file_list = [v08/'manifest.json',v08/'fixed_dev_split.csv',v08/'pretraining_selection.csv',v08/'development_results.csv',v08/'paired_controls.csv']
    real = raw.loc[raw.variant.eq('qmugs_matched'), ['seed','pr_auc']].rename(columns={'pr_auc':'real_dft'})
    assert len(real) == 5
    for shuffle,folder in SHUFFLES.items():
        folder_path = base / folder
        f = folder_path / 'development_results.csv'
        if not f.is_file():
            raise FileNotFoundError(f)
        file_list.append(f)
        df = pd.read_csv(f)
        d = df.loc[df.variant.eq('qmugs_matched_shuffled_dft'), ['seed','pr_auc']].rename(columns={'pr_auc':'shuffled_dft'})
        both = real.merge(d, on='seed', validate='one_to_one')
        if set(both.seed) != set(runs):
            raise RuntimeError(f'Unexpected training seeds for shuffle {shuffle}')
        both['shuffle_seed'] = shuffle
        both['delta_pr_auc'] = both.real_dft - both.shuffled_dft
        comparisons.append(both)
    all_comp = pd.concat(comparisons,ignore_index=True)
    by_seed = all_comp.groupby('seed')['delta_pr_auc'].agg(['mean','min','max','count'])
    mean_seed = by_seed['mean'].to_numpy()
    stats = {
        'status':'EXPLORATORY DEVELOPMENT ONLY, not external validation',
        'n_training_seeds':len(runs),
        'n_shuffle_maps':len(SHUFFLES),
        'n_correlated_comparisons':len(all_comp),
        'positive_comparisons':int((all_comp.delta_pr_auc > 0).sum()),
        'per_seed_means_positive':int((mean_seed > 0).sum()),
        'mean_per_seed_delta_pr_auc':float(np.mean(mean_seed)),
        'sample_sd_per_seed_delta_pr_auc':float(np.std(mean_seed,ddof=1)),
        'fixed_development_split_counts':counts,
        'reference_manifest_sha256':digest(v08/'manifest.json'),
        'selected_qmugs_source_counts':sel.source.value_counts().to_dict(),
        'caveat':'Five training seeds share same development evaluation; shuffled maps reused genuine baseline. Do not treat 15 contrasts as independent experiments.'
    }
    out.mkdir(parents=True)
    pd.DataFrame(rows if rows else raw).to_csv(out/'model_results_v08.csv', index=False)
    paired.to_csv(out/'paired_v08.csv', index=False)
    all_comp.to_csv(out/'three_shuffle_deltas.csv', index=False)
    by_seed.to_csv(out/'shuffle_deltas_by_seed.csv')
    pd.DataFrame([{'source_file':str(f), 'sha256':digest(f)} for f in file_list]).to_csv(out/'file_sha256.csv',index=False)
    (out/'audit.json').write_text(json.dumps(stats,indent=2,sort_keys=True),encoding='utf-8')
    print('RESULT: DEVELOPMENT AUDIT ONLY; test data not read')
    print('Training seeds:',runs)
    print('Fixed development split:',counts)
    print('Positive real-minus-shuffle contrasts:',stats['positive_comparisons'],'/',stats['n_correlated_comparisons'])
    print('Mean of five per-seed deltas:',round(stats['mean_per_seed_delta_pr_auc'],5))
    print('SAVED:',out)

if __name__ == '__main__':
    main()
