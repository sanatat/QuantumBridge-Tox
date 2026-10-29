#!/usr/bin/env python3
"""Strict external candidate identity/provenance audit, without model scoring.

Requires *manually curated* single AhR agonist-like assay binary labels.
Does NOT construct labels from unreviewed raw assay data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

REQ = ['smiles','label','assay_id','source','source_document','activity_direction','species']

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--tox',default='data/tox21_clean.csv.gz')
    ap.add_argument('--qmugs-pool',default='data/qmugs/qmugs_v06_pool.csv.gz')
    ap.add_argument('--out',required=True)
    ap.add_argument('--max-atoms',type=int,default=128)
    a=ap.parse_args()
    out=Path(a.out)
    if out.exists():
        raise SystemExit(f'Output already exists: {out}. Choose a fresh --out.')
    x=pd.read_csv(a.input,dtype={'label':str})
    missing=set(REQ)-set(x.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    for c in ('assay_id','source','source_document','activity_direction','species'):
        x[c]=x[c].astype('string').str.strip()
        if x[c].isna().any() or x[c].eq('').any() or x[c].str.lower().eq('nan').any():
            raise ValueError(f'{c} must be present on every record')
    for c in ('assay_id','activity_direction','species'):
        if x[c].nunique()!=1:
            raise ValueError(f'Primary cohort must use a single homogeneous {c}; found: {list(x[c].unique())[:12]}')
    direction=x.activity_direction.iloc[0].lower()
    if direction not in ('agonist','activation','agonist-like'):
        raise ValueError('Expected manually curated AhR agonist-like activation; binding or antagonism cannot be treated as identical to Tox21 NR-AhR')
    if x.assay_id.str.contains('tox21',case=False,na=False).any():
        raise ValueError('Tox21 assay origin flagged by assay_id; not an independent external cohort')
    starting=len(x)
    x['label']=x.label.astype('string').str.strip()
    valid=x.label.isin(['0','1'])
    bad_labels=int((~valid).sum()); x=x.loc[valid].copy()
    x['label']=x.label.astype(int)
    parsed=[]
    failures={'invalid_structure':0,'multi_fragment':0,'too_many_atoms':0}
    for _,r in x.iterrows():
        try:
            m=Chem.MolFromSmiles(str(r.smiles))
            if m is None:
                failures['invalid_structure']+=1;continue
            if len(Chem.GetMolFrags(m)) != 1:
                failures['multi_fragment']+=1;continue
            if m.GetNumAtoms() > a.max_atoms:
                failures['too_many_atoms']+=1;continue
            s=Chem.MolToSmiles(m,isomericSmiles=True)
            k=Chem.MolToInchiKey(m)
            scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            failures['invalid_structure']+=1;continue
        rec=r.to_dict()
        rec.update({'canonical_smiles':s,'inchikey':k,'scaffold':scaffold,'heavy_atoms':m.GetNumHeavyAtoms()})
        parsed.append(rec)
    y=pd.DataFrame(parsed)
    if y.empty:
        raise ValueError('No valid structures to audit')
    conflict_keys=y.groupby('inchikey').label.nunique()
    conflicted=set(conflict_keys[conflict_keys>1].index)
    conflicts=int(y.inchikey.isin(conflicted).sum())
    y=y[~y.inchikey.isin(conflicted)].copy()
    duplicate_same=int(y.inchikey.duplicated().sum())
    y=y.drop_duplicates('inchikey').copy()
    tox=pd.read_csv(a.tox,usecols=['inchikey','scaffold'])
    tox_keys=set(tox.inchikey.dropna().astype(str))
    # Scaffold overlap is descriptive. Acyclic scaffolds often equal empty string; ignore blanks.
    tox_scaf=set(tox.scaffold.dropna().astype(str)) - {''}
    pool=pd.read_csv(a.qmugs_pool,usecols=['inchikey'])
    qmugs_keys=set(pool.inchikey.dropna().astype(str))
    y['tox21_identity_overlap']=y.inchikey.isin(tox_keys)
    y['qmugs_pool_identity_overlap']=y.inchikey.isin(qmugs_keys)
    y['tox21_scaffold_overlap']=y.scaffold.fillna('').astype(str).apply(lambda t:bool(t) and t in tox_scaf)
    rejected_tox=int(y.tox21_identity_overlap.sum())
    rejected_qmugs=int((~y.tox21_identity_overlap & y.qmugs_pool_identity_overlap).sum())
    y=y.loc[~(y.tox21_identity_overlap | y.qmugs_pool_identity_overlap)].copy()
    y=y.sort_values('inchikey')
    n_pos=int(y.label.sum());n_neg=int(len(y)-n_pos)
    audit={
        'status':'CANDIDATE SCREENING ONLY, no model evaluated',
        'input':str(Path(a.input).resolve()),'input_sha256':sha256(a.input),
        'tox_input_sha256':sha256(a.tox),'qmugs_pool_sha256':sha256(a.qmugs_pool),
        'source':sorted(x.source.unique().tolist()),'assay_id':str(x.assay_id.iloc[0]),
        'activity_direction':direction,'species':str(x.species.iloc[0]),
        'input_rows':starting,'invalid_binary_labels':bad_labels,
        'invalid_structure':failures['invalid_structure'],'multi_fragment':failures['multi_fragment'],
        'too_many_atoms':failures['too_many_atoms'],'conflicted_label_rows':conflicts,
        'same_label_duplicate_rows':duplicate_same,
        'tox21_molecular_overlap_excluded':rejected_tox,
        'qmugs_pretraining_pool_overlap_excluded':rejected_qmugs,
        'retained_unique_nonoverlap':len(y),'positives':n_pos,'negatives':n_neg,
        'positive_prevalence':float(n_pos/len(y)) if len(y) else None,
        'remaining_with_Tox21_scaffold_overlap':int(y.tox21_scaffold_overlap.sum()),
        'sufficient_for_primary_metrics':bool(n_pos>=20 and n_neg>=20),
        'rdkit_version':rdkit.__version__,
        'warning':('QMugs exclusion covers the 32k project pool only; does not ensure no exposure to all full QMugs. '
                   'Chemical identity nonoverlap does not ensure assay provenance independence. '
                   'ToxCast includes Tox21 source data; assay metadata must be manually audited.')
    }
    out.mkdir(parents=True)
    y.to_csv(out/'curated_nonoverlap.csv',index=False)
    audit['curated_nonoverlap_sha256']=sha256(out/'curated_nonoverlap.csv')
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2))
    if not audit['sufficient_for_primary_metrics']:
        print('STOP: too few positive or negative cases for prespecified external metrics.')
    else:
        print('CANDIDATE PASSED AUTOMATED FILTERS. MANUAL ASSAY-PROVENANCE APPROVAL STILL REQUIRED.')

if __name__=='__main__':
    main()
