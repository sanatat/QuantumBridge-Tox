#!/usr/bin/env python3
"""Stream QMugs summary.csv into a reproducible, bounded-memory molecular pool.

One representative conformer per CHEMBL id: conf_00 only, NOT necessarily
lowest-energy conformer. Dataset MUST be downloaded separately by the user.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')
COLUMNS = ['chembl_id', 'conf_id', 'smiles', 'heavy_atoms',
           'DFT_DIPOLE_TOT', 'DFT_HOMO_ENERGY', 'DFT_LUMO_ENERGY',
           'significant_negative_wavenumbers']


def hash_sample(value, rate):
    n = int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), 'big')
    return n < int(rate * (2**64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='data/qmugs/summary.csv')
    ap.add_argument('--tox', default='data/tox21_clean.csv.gz')
    ap.add_argument('--out', default='data/qmugs/qmugs_v06_pool.csv.gz')
    ap.add_argument('--audit-out', default='data/qmugs/qmugs_v06_audit.json')
    ap.add_argument('--sample-rate', type=float, default=0.05)
    ap.add_argument('--chunksize', type=int, default=20000)
    ap.add_argument('--max-heavy-atoms', type=int, default=100)
    ap.add_argument('--keep-imaginary', action='store_true', help='Include conformers flagged with significant negative frequencies')
    args = ap.parse_args()
    if not (0 < args.sample_rate <= 1) or args.chunksize < 100:
        ap.error('sample-rate must be (0,1]; chunksize >= 100')
    for p in (args.source, args.tox):
        if not Path(p).is_file():
            ap.error(f'Missing file {p}')
    if Path(args.out).exists() or Path(args.audit_out).exists():
        ap.error('Output or audit already exists; select NEW paths to protect prior results')
    tox_keys = set(pd.read_csv(args.tox, usecols=['inchikey']).inchikey.dropna())
    rows_by_key = {}
    seen_chembl = set()
    counts = dict(scanned_rows=0, conf00_rows=0, selected_by_hash=0,
                  missing_invalid=0, nonconverged_filtered=0, discarded_heavy=0,
                  excluded_tox21=0, repeated_inchikey=0, accepted=0)
    iterator = pd.read_csv(args.source, usecols=COLUMNS, chunksize=args.chunksize,
                           low_memory=False)
    for k, chunk in enumerate(iterator, start=1):
        counts['scanned_rows'] += len(chunk)
        chunk = chunk.loc[chunk.conf_id.eq('conf_00')]
        counts['conf00_rows'] += len(chunk)
        chunk = chunk.loc[chunk.chembl_id.map(lambda x: hash_sample(x, args.sample_rate))]
        counts['selected_by_hash'] += len(chunk)
        for r in chunk.itertuples(index=False):
            if not isinstance(r.chembl_id,str) or r.chembl_id in seen_chembl:
                continue
            seen_chembl.add(r.chembl_id)
            if not args.keep_imaginary and str(r.significant_negative_wavenumbers).strip().lower() in ('true','1','yes'):
                counts['nonconverged_filtered'] += 1
                continue
            if not (isinstance(r.smiles, str) and r.smiles):
                counts['missing_invalid'] += 1
                continue
            props = np.array([r.DFT_DIPOLE_TOT, r.DFT_HOMO_ENERGY, r.DFT_LUMO_ENERGY], dtype=float)
            if not np.isfinite(props).all():
                counts['missing_invalid'] += 1
                continue
            m = Chem.MolFromSmiles(r.smiles)
            if m is None or m.GetNumAtoms() == 0:
                counts['missing_invalid'] += 1
                continue
            if m.GetNumHeavyAtoms() > args.max_heavy_atoms:
                counts['discarded_heavy'] += 1
                continue
            # A single disconnected input can be salts/multiple fragments and
            # is less interpretable under 2D molecular graph pooling.
            if len(Chem.GetMolFrags(m)) != 1:
                counts['missing_invalid'] += 1
                continue
            ik = Chem.MolToInchiKey(m)
            if ik in tox_keys:
                counts['excluded_tox21'] += 1
                continue
            if ik in rows_by_key:
                counts['repeated_inchikey'] += 1
                continue
            rows_by_key[ik] = {
                'chembl_id':r.chembl_id, 'conf_id':'conf_00',
                'inchikey': ik, 'smiles': Chem.MolToSmiles(m, isomericSmiles=True),
                'mu':float(props[0]), 'homo':float(props[1]),'lumo':float(props[2]),
                'heavy_atoms':int(m.GetNumHeavyAtoms()),
                'mw':float(Descriptors.MolWt(m)),
                'logp':float(Crippen.MolLogP(m)),
                'aromatic_rings':int(rdMolDescriptors.CalcNumAromaticRings(m)),
                'halogen': int(any(a.GetSymbol() in {'Cl','Br','I'} for a in m.GetAtoms()))
            }
        if k % 25 == 0:
            print(f'chunks={k}, rows={counts["scanned_rows"]:,}, pool={len(rows_by_key):,}', flush=True)
    counts['accepted'] = len(rows_by_key)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows_by_key.values())).sort_values('inchikey').to_csv(out,index=False,compression='gzip')
    meta = {'protocol':'one conf_00 per ChEMBL id; hashed subsample; unique full InChIKey; excludes all Tox21; single-component molecules',
            'sample_rate':args.sample_rate,'max_heavy_atoms':args.max_heavy_atoms,
            'kept_significant_negative_wavenumbers':args.keep_imaginary,
            'dft_labels':['DFT_DIPOLE_TOT','DFT_HOMO_ENERGY','DFT_LUMO_ENERGY'],
            'note':'GFN2_POLARIZABILITY_MOLECULAR is NOT a DFT target and has been excluded',
            'counts':counts,'output':str(out)}
    Path(args.audit_out).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
    if len(rows_by_key) < 6000:
        print('WARNING: pool smaller than 6000; consider new paths and a larger --sample-rate.',flush=True)


if __name__ == '__main__':
    main()
