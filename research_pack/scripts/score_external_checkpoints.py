#!/usr/bin/env python3
"""Score manually approved, non-overlapping external AhR cohort with fixed v0.8 GNNs.

No GNN retraining or score-based selection. Optional RF is deterministically
reconstructed on exactly the fixed inner Tox21 train split and verified against
saved v0.8 development probabilities before external prediction.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED = {'inchikey','canonical_smiles','label','scaffold','assay_id','source'}
GNNS = ('qmugs_matched','qmugs_matched_rdkit','scratch')

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()

def fingerprint_matrix(smiles):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    rows=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f'Invalid fingerprint input {s}')
        vec=np.zeros((1024,),dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),vec)
        rows.append(vec)
    return np.stack(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True, help='Curated output of external_cohort_audit.py')
    p.add_argument('--v08',default='results/gnn_v08_fixed_split_5k')
    p.add_argument('--out',required=True)
    p.add_argument('--approve-assay-id',required=True,help='Must match manually checked assay_id exactly')
    p.add_argument('--include-rf',action='store_true',help='Reconstruct RF and verify saved development predictions')
    p.add_argument('--bootstrap-reps',type=int,default=2000)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists():
        raise SystemExit('Refusing to overwrite existing output folder; choose a new --out')
    source=Path(a.input)
    audit_file=source.parent/'audit.json'
    if not audit_file.exists():
        raise ValueError('No audit.json beside cohort: run external_cohort_audit.py first')
    audit=json.loads(audit_file.read_text())
    if not audit.get('sufficient_for_primary_metrics'):
        raise ValueError('Cohort failed minimum positive/negative class rule')
    if 'curated_nonoverlap_sha256' in audit and audit['curated_nonoverlap_sha256']!=digest(source):
        raise ValueError('Audited cohort file changed after audit!')
    df=pd.read_csv(source)
    if not REQUIRED.issubset(df.columns):
        raise ValueError(f'Missing columns: {REQUIRED-set(df.columns)}')
    if df.inchikey.duplicated().any() or df.inchikey.isna().any() or not df.label.isin([0,1]).all():
        raise ValueError('Invalid unique binary external cohort')
    if df.assay_id.nunique()!=1 or str(df.assay_id.iloc[0])!=a.approve_assay_id:
        raise ValueError('Explicit assay approval does not match the audited cohort assay_id')
    if len(df)<40 or int(df.label.sum())<20 or int((df.label==0).sum())<20:
        raise ValueError('Too few evaluated cases in a class; do not claim external validation')
    v08=Path(a.v08)
    manifest=json.loads((v08/'manifest.json').read_text())
    if manifest.get('status') != 'DEVELOPMENT ONLY' or manifest.get('version')!='v0.8':
        raise ValueError('Unexpected model protocol')
    seeds=manifest['parameters']['seeds']
    if not all(x in manifest['parameters']['variants'] for x in GNNS):
        raise ValueError('Required reference variants were not trained')
    if not (v08/'checkpoints').is_dir():
        raise ValueError('Missing saved GNN checkpoints. Re-running on the existing result directory is forbidden')
    code_path=Path.cwd().resolve()
    for fname,expected in manifest['code_sha256'].items():
        check=code_path/fname
        if not check.exists() or digest(check)!=expected:
            raise ValueError(f'Original model source code differs from frozen manifest: {fname}')
    sys.path.insert(0,str(code_path))
    import gnn_quantumbridge as base
    torch.set_num_threads(a.threads)
    max_atoms=int(manifest['parameters']['max_atoms'])
    graphs=[]
    for smi in df.canonical_smiles:
        g=base.to_graph(smi,max_atoms=max_atoms)
        if g is None:
            raise ValueError('External molecule invalid for frozen GNN; audit/model graph mismatch')
        graphs.append((g,torch.tensor([0.0],dtype=torch.float32)))
    y=df.label.to_numpy(dtype=int)
    preds=df[['inchikey','assay_id','label','scaffold']].copy()
    scores=[]
    models=list(GNNS)
    summary_by_seed={}
    for seed in seeds:
        for variant in GNNS:
            file=v08/'checkpoints'/f'{variant}_seed{seed}.pt'
            if not file.is_file():
                raise FileNotFoundError(file)
            state=torch.load(file,map_location='cpu',weights_only=True)
            model=base.MolecularGNN(hidden=int(manifest['parameters']['hidden']),output=1)
            model.load_state_dict(state, strict=True)
            model.eval()
            raw=base.predict(model,graphs,batch_size=int(manifest['parameters']['batch'])).ravel()
            # sigmoid of logits computed with numpy, clipped for numerical stability
            prob=1/(1+np.exp(-np.clip(raw,-70,70)))
            preds[f'{variant}_seed{seed}']=prob
            scores.append({'seed':seed,'variant':variant,'pr_auc':float(average_precision_score(y,prob)),
                'roc_auc':float(roc_auc_score(y,prob)), 'n':len(y), 'positives':int(y.sum())})
        summary_by_seed[seed]={}
        print(f'SCORED seed={seed} frozen GNNs on {len(y)} external assay observations',flush=True)
    if a.include_rf:
        split=pd.read_csv(v08/'fixed_dev_split.csv')
        tr=split.assignment.eq('inner_train').to_numpy()
        if not all(c in split.columns for c in ['canonical_smiles',manifest['parameters']['task']]):
            raise ValueError('Fixed split training labels/smiles missing')
        gen_x=fingerprint_matrix(split.canonical_smiles)
        ext_x=fingerprint_matrix(df.canonical_smiles)
        target=manifest['parameters']['task']
        train_y=split[target].to_numpy(dtype=int)
        dev_keys=split.loc[split.assignment.eq('outer_development_eval'),'inchikey'].to_numpy()
        for seed in seeds:
            rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,
                 class_weight='balanced_subsample',n_jobs=a.threads,random_state=seed)
            rf.fit(gen_x[tr],train_y[tr])
            p_class=list(rf.classes_).index(1)
            prior=pd.read_csv(v08/f'predictions_seed{seed}.csv')
            if list(prior.inchikey) != list(dev_keys):
                raise RuntimeError('RF verification failed due to development row identity mismatch')
            check=rf.predict_proba(gen_x[split.assignment.eq('outer_development_eval')])[:,p_class]
            maxdiff=float(np.max(np.abs(check-prior.random_forest.to_numpy())))
            if maxdiff > 1e-8:
                raise RuntimeError(f'RF refit does not reproduce saved dev predictions, seed={seed}, max delta={maxdiff:.3g}')
            prob=rf.predict_proba(ext_x)[:,p_class]
            preds[f'random_forest_seed{seed}']=prob
            scores.append({'seed':seed,'variant':'random_forest',
                'pr_auc':float(average_precision_score(y,prob)),
                'roc_auc':float(roc_auc_score(y,prob)), 'n':len(y),'positives':int(y.sum())})
            print(f'VERIFIED and scored RF seed={seed} (max saved-dev diff {maxdiff:.2g})',flush=True)
        models.append('random_forest')
    out.mkdir(parents=True)
    pd.DataFrame(scores).to_csv(out/'external_scores.csv',index=False)
    preds.to_csv(out/'external_predictions.csv',index=False)
    pairs=[('qmugs_matched','qmugs_matched_rdkit','matched DFT minus matched RDKit'),
        ('qmugs_matched','scratch','matched DFT minus scratch')]
    if a.include_rf:
        pairs.append(('qmugs_matched','random_forest','matched DFT minus RF'))
    rows=[]
    # Paired *scaffold cluster* resampling on the external cohort; repeated models share specimens.
    scaf=preds.scaffold.fillna('').astype(str).to_numpy()
    groups=np.array([s if s else 'ACYCLIC_'+str(k) for k,s in zip(preds.inchikey,scaf)])
    uniq,inv=np.unique(groups,return_inverse=True)
    clusters=[np.flatnonzero(inv==j) for j in range(len(uniq))]
    for seed in seeds:
        for model_a,model_b,label in pairs:
            pa=preds[f'{model_a}_seed{seed}'].to_numpy(dtype=float)
            pb=preds[f'{model_b}_seed{seed}'].to_numpy(dtype=float)
            observed=float(average_precision_score(y,pa)-average_precision_score(y,pb))
            rng=np.random.default_rng(np.random.SeedSequence([20261029,int(seed),len(label)]))
            samples=[]
            for _ in range(a.bootstrap_reps):
                for _attempt in range(100):
                    idx=np.concatenate([clusters[k] for k in rng.integers(len(clusters),size=len(clusters))])
                    if len(np.unique(y[idx]))==2:
                        break
                else:
                    raise RuntimeError('External class balance too sparse for valid scaffold cluster resampling')
                samples.append(average_precision_score(y[idx],pa[idx])-average_precision_score(y[idx],pb[idx]))
            lo,hi=np.quantile(samples,[0.025,0.975])
            rows.append({'seed':seed,'contrast':label,'observed_delta_pr_auc':observed,
                'ci_2p5':float(lo),'ci_97p5':float(hi),
                'interval_excludes_zero':bool(lo>0 or hi<0),
                'bootstrap_unit':'external Murcko scaffold (acyclic cases individually)',
                'bootstrap_reps':a.bootstrap_reps})
    pd.DataFrame(rows).to_csv(out/'external_paired_bootstrap.csv',index=False)
    receipt={'status':'EXTERNAL EVALUATION OF USER-APPROVED CURATED COHORT',
             'assay_id':a.approve_assay_id,'external_sha256':digest(source),
             'audit_sha256':digest(audit_file),'v08_manifest_sha256':digest(v08/'manifest.json'),
             'n':len(y),'positives':int(y.sum()),'negative':int((y==0).sum()),
             'prevalence_baseline_average_precision':float(np.mean(y)),
             'models':models,'initialization_seeds':seeds,
             'scientific_warning':'External comparability and independent experimental source require human verification; metrics are not automatically equivalent to Tox21 NR-AhR.'}
    (out/'evaluation_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(pd.DataFrame(scores).groupby('variant').agg(mean_pr=('pr_auc','mean'),sd_pr=('pr_auc','std')).round(4).to_string())
    print('SAVED:',out)

if __name__=='__main__':
    main()
