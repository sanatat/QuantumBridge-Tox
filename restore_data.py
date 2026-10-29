#!/usr/bin/env python3
"""Restore missing public Tox21/QM9 files and merge by molecular InChIKey.

Requires numpy, pandas, RDKit and scikit-learn from requirements.txt.
Does not overwrite existing inputs, datasets or result files.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
QM9_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
TOX21_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
DFT_COLUMNS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298']


def download_if_missing(url, target):
    if target.is_file() and target.stat().st_size > 0:
        print(f'PRESERVED: {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.partial')
    print(f'DOWNLOADING: {url}', flush=True)
    try:
        with urllib.request.urlopen(url, timeout=90) as stream, open(part, 'wb') as f:
            shutil.copyfileobj(stream, f)
        if part.stat().st_size < 1000:
            raise ValueError('Downloaded file suspiciously small')
        part.replace(target)
        print(f'SAVED: {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)', flush=True)
    except Exception:
        print(f'Download failed. Check network; you can manually save to {target}', file=sys.stderr)
        raise


def to_inchikey(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def merge():
    target = RESULTS / 'qm9_tox21_dft_merged.csv'
    if target.is_file() and target.stat().st_size:
        print(f'PRESERVED: {target.relative_to(ROOT)}', flush=True)
        return
    qm = pd.read_csv(DATA / 'qm9.csv', low_memory=False)
    tox = pd.read_csv(DATA / 'tox21_clean.csv.gz', low_memory=False)
    if 'smiles' not in qm or 'inchikey' not in tox:
        raise ValueError('Input tables do not contain expected smiles/inchikey columns')
    cols = [c for c in DFT_COLUMNS if c in qm.columns]
    if not cols:
        raise ValueError(f'No expected DFT columns. Available: {list(qm.columns)}')
    qm['inchikey'] = qm['smiles'].map(to_inchikey)
    qm = qm.dropna(subset=['inchikey'])
    print(f'QM9 duplicated InChIKeys: {qm.inchikey.duplicated().sum():,}', flush=True)
    print(f'Tox21 duplicated InChIKeys: {tox.inchikey.duplicated().sum():,}', flush=True)
    varied = qm.groupby('inchikey')[cols].nunique(dropna=True).gt(1).any(axis=1)
    print(f'QM9 InChIKeys with differing DFT values: {int(varied.sum()):,}', flush=True)
    print('WARNING: Duplicate QM9 structures retain FIRST DFT record (exploratory only).', flush=True)
    qm = qm.drop_duplicates('inchikey', keep='first')
    tox = tox.drop_duplicates('inchikey', keep='first')
    merged = tox.merge(qm[['inchikey', *cols]], on='inchikey', validate='one_to_one', how='inner')
    RESULTS.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False)
    print(f'MERGED: {len(merged)} unique structures; DFT fields={cols}', flush=True)
    print(f'SAVED: {target.relative_to(ROOT)}', flush=True)
    for task in ('NR-AhR', 'NR-ER', 'SR-p53'):
        if task not in merged.columns:
            continue
        y = merged[task].dropna()
        print(f'  {task}: labeled={len(y)}, positive={int((y==1).sum())}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download-only', action='store_true', help='Fetch missing public inputs only')
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    download_if_missing(TOX21_URL, DATA / 'tox21.csv.gz')
    download_if_missing(QM9_URL, DATA / 'qm9.csv')
    if args.download_only:
        return
    if not (DATA / 'tox21_clean.csv.gz').is_file():
        print('Preparing standardized Tox21 ...', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'quantumbridge.py'), 'prepare', '--out', str(DATA)], check=True, cwd=ROOT)
    else:
        print('PRESERVED: data/tox21_clean.csv.gz', flush=True)
    merge()
    if not (RESULTS / 'baseline.json').is_file():
        print('Reproducing Tox21 SR-p53 baseline ...', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'quantumbridge.py'), 'baseline', '--task', 'SR-p53', '--jobs', '2', '--report', str(RESULTS / 'baseline.json')], check=True, cwd=ROOT)
    else:
        print('PRESERVED: results/baseline.json', flush=True)
    print('\nDONE. Comparison file can be regenerated with: python compare_dft.py', flush=True)
    print('For scientific interpretation, check DFT duplicate chemistry and scaffold-label imbalance.')


if __name__ == '__main__':
    main()
