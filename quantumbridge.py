#!/usr/bin/env python3
"""QuantumBridge-Tox: Tox21 preparation, CPU baseline, and external DFT overlap.

No DFT calculations are performed; external DFT annotations require a separate dataset.
"""
import argparse
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
RDLogger.DisableLog('rdApp.*')
URL='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
TASKS=['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

def standardize(s):
    try:
        m=Chem.MolFromSmiles(str(s))
        if m is None: return None,None,None
        # Keep largest covalent fragment and neutralize without changing labels' chemical identity.
        fragments=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True)
        m=max(fragments,key=lambda x:x.GetNumHeavyAtoms())
        cs=Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
        key=inchi.MolToInchiKey(m)
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m) or cs
        return cs,key,scaffold
    except Exception: return None,None,None

def prepare(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    source=Path(args.input) if args.input else root/'tox21.csv.gz'
    if not source.exists():
        print('Downloading original Tox21 dataset:',URL)
        try: urllib.request.urlretrieve(URL,source)
        except Exception as e: raise SystemExit(f'Download failed: {e}\nManually download {URL} and rerun --input FILE')
    df=pd.read_csv(source)
    if 'smiles' not in df: raise SystemExit('SMILES column not found')
    data=[standardize(s) for s in df['smiles']]
    df[['canonical_smiles','inchikey','scaffold']]=pd.DataFrame(data,index=df.index)
    df=df.dropna(subset=['canonical_smiles','inchikey']).copy()
    # Replicates may have contradictory labels: preserve all for audit, and use grouped molecular splits.
    df.to_csv(root/'tox21_clean.csv.gz',index=False,compression='gzip')
    summary={'total_valid_rows':int(len(df)),'unique_inchikeys':int(df.inchikey.nunique()),'duplicate_molecular_rows':int(len(df)-df.inchikey.nunique()),'task_label_counts':{k:int(df[k].notna().sum()) for k in TASKS if k in df}}
    (root/'tox21_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

def fingerprint(s,n_bits=1024):
    mol=Chem.MolFromSmiles(s)
    fp=AllChem.GetMorganGenerator(radius=2,fpSize=n_bits).GetFingerprint(mol)
    arr=np.zeros(n_bits,dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

def baseline(args):
    src=Path(args.data)
    if not src.exists(): raise SystemExit(f'Missing {src}; run prepare first')
    df=pd.read_csv(src).drop_duplicates('inchikey').reset_index(drop=True)
    task=args.task
    if task not in TASKS: raise SystemExit('Unknown task')
    df=df[df[task].notna()].copy().reset_index(drop=True)
    if args.max_rows: df=df.iloc[:args.max_rows].copy()
    if len(df)<30: raise SystemExit('Too few labeled rows')
    X=np.stack([fingerprint(s) for s in df.canonical_smiles]); y=df[task].astype(int).to_numpy()
    groups=df['scaffold'].fillna(df['inchikey']).values
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    train_idx,test_idx=next(splitter.split(X,y,groups))
    train_X,train_y=X[train_idx],y[train_idx]; test_X,test_y=X[test_idx],y[test_idx]
    model=RandomForestClassifier(n_estimators=args.trees,max_depth=15,min_samples_leaf=2,n_jobs=args.jobs,class_weight='balanced_subsample',random_state=42)
    if len(set(train_y))<2: raise SystemExit('Training split has only one class; use another task or more rows')
    model.fit(train_X,train_y)
    proba=model.predict_proba(test_X)[:,list(model.classes_).index(1)]
    result={'task':task,'split':'scaffold-grouped','training_rows':len(train_idx),'test_rows':len(test_idx),'test_positives':int(test_y.sum()),'test_negatives':int((1-test_y).sum()),'roc_auc':float(roc_auc_score(test_y,proba)) if len(set(test_y))>1 else None,'pr_auc':float(average_precision_score(test_y,proba)) if len(set(test_y))>1 else None,'note':'Single split baseline; not GNN or DFT-enhanced model.'}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

def overlap(args):
    tox=pd.read_csv(args.tox,dtype={'inchikey':str})
    if 'inchikey' not in tox: raise SystemExit('Run prepare first')
    known=set(tox.inchikey.dropna()); found=set(); seen=0
    p=Path(args.dft)
    if not p.exists(): raise SystemExit(f'Missing DFT file: {p}')
    for chunk in pd.read_csv(p,chunksize=25000,low_memory=False):
        seen+=len(chunk)
        if args.key_column in chunk:
            keys=chunk[args.key_column].dropna().astype(str)
        elif args.smiles_column in chunk:
            keys=chunk[args.smiles_column].map(lambda s:standardize(s)[1]).dropna()
        else: raise SystemExit(f'DFT CSV needs {args.key_column!r} or {args.smiles_column!r} column. Columns: {list(chunk.columns)}')
        found.update(known.intersection(keys))
    matched=tox[tox.inchikey.isin(found)]
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    matched.to_csv(args.out,index=False)
    print(json.dumps({'dft_rows_scanned':seen,'tox21_unique_molecules':len(known),'overlap_unique_molecules':len(found),'matched_tox21_rows':len(matched),'out':args.out},indent=2))
    print('WARNING: matching by InChIKey does not ensure same protonation/tautomer/DFT conformer. Review chemistry and DFT settings before modeling.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--input');a.add_argument('--out',default='data');a.set_defaults(func=prepare)
    a=sub.add_parser('baseline');a.add_argument('--data',default='data/tox21_clean.csv.gz');a.add_argument('--task',default='SR-p53');a.add_argument('--max-rows',type=int,default=0);a.add_argument('--trees',type=int,default=120);a.add_argument('--jobs',type=int,default=2);a.add_argument('--report',default='results/baseline.json');a.set_defaults(func=baseline)
    a=sub.add_parser('overlap');a.add_argument('--tox',default='data/tox21_clean.csv.gz');a.add_argument('--dft',required=True);a.add_argument('--key-column',default='inchikey');a.add_argument('--smiles-column',default='smiles');a.add_argument('--out',default='results/tox21_dft_matches.csv');a.set_defaults(func=overlap)
    args=ap.parse_args();args.func(args)
