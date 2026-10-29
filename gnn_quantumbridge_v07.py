#!/usr/bin/env python3
"""Development-only v0.7: matched-chemistry DFT-versus-RDKit controls for Tox21 NR-AhR.

Reuses UNMODIFIED gnn_quantumbridge.py (v0.3) and
 gnn_quantumbridge_v05.py (v0.5); no original holdout test evaluation.
Model architecture, pretraining targets, batch, seed and toxicity development
split are held fixed across data sources. QMugs versus QM9 DFT protocols differ;
differences cannot be attributed solely to molecular domain shift.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler

import gnn_quantumbridge as base
import gnn_quantumbridge_v05 as v5

DFT3 = ['mu', 'homo', 'lumo']
VARIANTS = ['scratch', 'qm9_dft', 'qmugs_random', 'qmugs_matched', 'qmugs_rdkit', 'qmugs_matched_rdkit']
MATCH_FEATURES = ['heavy_atoms','mw','logp','aromatic_rings']


def get_tox_train_features(dev, train_idx):
    rows=[]
    for s in dev.iloc[train_idx].canonical_smiles:
        mol=Chem.MolFromSmiles(s)
        rows.append([float(mol.GetNumHeavyAtoms()), float(Descriptors.MolWt(mol)),
                     float(Crippen.MolLogP(mol)),float(rdMolDescriptors.CalcNumAromaticRings(mol))])
    return np.asarray(rows,dtype=np.float64)


def selected_qmugs(pool, val_idx, seed, dev, train_idx, n):
    cand = pool.drop(index=val_idx)
    if len(cand) < n:
        raise ValueError(f'QMugs candidate pool {len(cand)} < requested {n}')
    # Matching depends ONLY on inner training set chemistry, not labels,
    # outer development evaluation or frozen prior test.
    train_features=get_tox_train_features(dev,train_idx)
    scaler=RobustScaler().fit(train_features)
    tox_scaled=np.clip(scaler.transform(train_features),-20,20)
    pool_scaled=np.clip(scaler.transform(cand[MATCH_FEATURES].to_numpy()),-20,20)
    distances,_=cKDTree(tox_scaled).query(pool_scaled,k=1,workers=1)
    c=cand.copy()
    c['match_distance']=distances
    # Stable determinism independent of CSV sorting.
    c=c.sort_values(['match_distance','inchikey'],kind='mergesort')
    matched=c.head(n).drop(columns=['match_distance'])
    random_set=cand.sample(n=n,random_state=seed+31045)
    return random_set,matched


def build_items(df, targets, max_atoms, rdkit=False):
    result=[]
    for r in df.itertuples(index=False):
        g=base.to_graph(r.smiles,max_atoms=max_atoms)
        if g is None:
            raise ValueError(f'Unexpected invalid graph in preselected pool: {r.smiles[:60]}')
        vals=v5.compute_rdkit_descriptors(r.smiles) if rdkit else np.array(
            [getattr(r,key) for key in targets],dtype=np.float32)
        result.append((g,torch.tensor(vals,dtype=torch.float32)))
    return result


def build_qm9(q, n):
    qm=q['qm']
    def make(indices):
        out=[]
        for ix in indices:
            ix=int(ix)
            g=q['items'][ix][0]
            y=torch.tensor(qm.iloc[ix][DFT3].to_numpy(dtype=np.float32),dtype=torch.float32)
            out.append((g,y))
        return out
    return make(q['train_ix'][:n]),make(q['val_ix'])


def selection_digest(df):
    """Stable fingerprint of the set of molecules used for pretraining."""
    keys = sorted(df.inchikey.astype(str).tolist())
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicated molecular identities in pretraining selection')
    return hashlib.sha256('\n'.join(keys).encode('utf-8')).hexdigest()


def paired_control_deltas(df):
    """Within-seed comparisons on precisely the same toxicity development split."""
    comparisons = [
        ('qmugs_matched', 'qmugs_matched_rdkit', 'matched DFT minus matched RDKit'),
        ('qmugs_random', 'qmugs_rdkit', 'random DFT minus random RDKit'),
        ('qmugs_matched', 'qmugs_random', 'matched DFT minus random DFT'),
        ('qmugs_matched_rdkit', 'qmugs_rdkit', 'matched RDKit minus random RDKit'),
        ('qmugs_matched', 'scratch', 'matched DFT minus scratch'),
        ('qmugs_matched', 'random_forest', 'matched DFT minus RF'),
    ]
    rows = []
    for a, b, name in comparisons:
        left = df[df.variant == a].set_index('seed')
        right = df[df.variant == b].set_index('seed')
        if set(left.index) != set(right.index):
            raise RuntimeError(f'Seed mismatch for comparison: {name}')
        for seed in sorted(left.index):
            r = {'seed': int(seed), 'contrast': name, 'model_a': a, 'model_b': b}
            for metric in ['roc_auc', 'pr_auc', 'f1', 'mcc']:
                r['delta_' + metric] = float(left.at[seed, metric] - right.at[seed, metric])
            rows.append(r)
    return pd.DataFrame(rows)


def audit_selection(df):
    return {'n':int(len(df)),
            'median_heavy_atoms':float(df.heavy_atoms.median()),
            'median_mw':float(df.mw.median()),
            'median_logp':float(df.logp.median()),
            'median_aromatic_rings':float(df.aromatic_rings.median()),
            'fraction_heavy_halogens':float(df.halogen.mean())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tox',default='data/tox21_clean.csv.gz')
    ap.add_argument('--qm9',default='data/qm9.csv')
    ap.add_argument('--qmugs-pool',default='data/qmugs/qmugs_v06_pool.csv.gz')
    ap.add_argument('--prior-split',default='results/gnn_v03_5k/tox_split.csv')
    ap.add_argument('--out',default='results/gnn_v07_dev')
    ap.add_argument('--task',default='NR-AhR')
    ap.add_argument('--seeds',type=v5.parse_ints,default=[42,123,2026])
    ap.add_argument('--n-pretrain',type=int,default=5000)
    ap.add_argument('--qm9-val-n',type=int,default=500)
    ap.add_argument('--qmugs-val-n',type=int,default=500)
    ap.add_argument('--qm9-sample-seed',type=int,default=42)
    ap.add_argument('--qmugs-sample-seed',type=int,default=42)
    ap.add_argument('--pretrain-epochs',type=int,default=8)
    ap.add_argument('--finetune-epochs',type=int,default=12)
    ap.add_argument('--patience',type=int,default=4)
    ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--max-atoms',type=int,default=128)
    ap.add_argument('--min-positives',type=int,default=12)
    ap.add_argument('--save-checkpoints',action='store_true')
    args=ap.parse_args()
    if args.n_pretrain<100 or min(args.qm9_val_n,args.qmugs_val_n)<20:
        ap.error('n-pretrain >=100, validation sizes >=20')
    if min(args.threads,args.batch,args.patience,args.pretrain_epochs,args.finetune_epochs)<1:
        ap.error('threads, batch, patience, epochs must be >=1')
    if not args.seeds or min(args.seeds)<0 or len(args.seeds)!=len(set(args.seeds)):
        ap.error('seeds must be a nonempty comma-separated list without duplicates')
    # v0.5 helpers require .n_qm9 and .mode; keep local to this driver.
    args.n_qm9=args.n_pretrain
    args.mode='multiseed'
    v5.require_files(args)
    if not Path(args.qmugs_pool).is_file():
        ap.error(f'QMugs pool missing: {args.qmugs_pool}; run extract_qmugs_v06.py first')
    out=v5.safe_output(args)
    torch.set_num_threads(args.threads)
    dev,dev_items,all_tox,heldout_n=v5.load_frozen_development(args)
    qm9=v5.prepare_qm9(args,all_tox)
    pool=pd.read_csv(args.qmugs_pool)
    required={'smiles','inchikey','mu','homo','lumo',*MATCH_FEATURES,'halogen'}
    if not required.issubset(pool):
        raise ValueError(f'QMugs pool missing columns: {required-set(pool)}')
    if pool.inchikey.isna().any() or pool.inchikey.duplicated().any():
        raise ValueError('QMugs pool must have unique nonempty InChIKeys')
    tox_keys=set(all_tox.inchikey)
    tox_keys |= set(pd.read_csv(args.tox,usecols=['inchikey']).inchikey.dropna())
    if tox_keys & set(pool.inchikey):
        raise ValueError('QMugs pool contains Tox21 molecules; re-run extractor')
    if len(pool)<args.n_pretrain+args.qmugs_val_n:
        raise ValueError('QMugs pool too small; extract at higher --sample-rate')
    # Fixed molecular validation partition, independent of tox labels.
    qval_idx=pool.sample(n=args.qmugs_val_n,random_state=args.qmugs_sample_seed).index
    qval=pool.loc[qval_idx].copy()
    rows=[]; pre_rows=[]; audit=[]; selection_rows=[]
    print(f'QMugs pool unique={len(pool)}, validation={len(qval)}, '
          f'QMugs DFT targets={DFT3} (NOT GFN2 polarizability)',flush=True)
    for seed in args.seeds:
        train_ix,val_ix,eval_ix=v5.development_split(dev,args,seed)
        assign=np.full(len(dev),'UNASSIGNED',dtype=object)
        assign[train_ix]='inner_train';assign[val_ix]='inner_earlystop_val';assign[eval_ix]='outer_development_eval'
        dev[['inchikey','canonical_smiles','scaffold',args.task]].assign(assignment=assign).to_csv(
            out/f'dev_split_seed{seed}.csv',index=False)
        train=[dev_items[int(i)] for i in train_ix]
        val=[dev_items[int(i)] for i in val_ix]
        evaluation=[dev_items[int(i)] for i in eval_ix]
        rf_res,rf_prob=v5.make_rf(dev,train_ix,val_ix,eval_ix,args,seed)
        rows.append({'seed':seed,'variant':'random_forest','n_pretrain':0,
                     'n_eval':len(evaluation),**rf_res})
        predictions=pd.DataFrame({'inchikey':dev.iloc[eval_ix].inchikey.values,
                                  'label':dev.iloc[eval_ix][args.task].values,
                                  'random_forest':rf_prob})
        random_mugs,matched_mugs=selected_qmugs(pool,qval_idx,seed,dev,train_ix,args.n_pretrain)
        r_hash, m_hash = selection_digest(random_mugs), selection_digest(matched_mugs)
        # The paired control must see precisely the same molecular IDs, not only
        # the same descriptor distribution or random generator state.
        assert selection_digest(matched_mugs) == m_hash
        audit.extend([{'seed':seed,'variant':'qmugs_random','selection_sha256':r_hash,**audit_selection(random_mugs)},
                      {'seed':seed,'variant':'qmugs_matched','selection_sha256':m_hash,**audit_selection(matched_mugs)},
                      {'seed':seed,'variant':'qmugs_rdkit','selection_sha256':r_hash,**audit_selection(random_mugs)},
                      {'seed':seed,'variant':'qmugs_matched_rdkit','selection_sha256':m_hash,**audit_selection(matched_mugs)}])
        for selection_name,selected in [('qmugs_random',random_mugs),('qmugs_matched',matched_mugs)]:
            selection_rows.extend({'seed':seed,'source':selection_name,'inchikey':key} for key in selected.inchikey.tolist())
        sources={
          'qm9_dft':(lambda:build_qm9(qm9,args.n_pretrain),DFT3),
          'qmugs_random':(lambda: (build_items(random_mugs,DFT3,args.max_atoms),
                                  build_items(qval,DFT3,args.max_atoms)), DFT3),
          'qmugs_matched':(lambda:(build_items(matched_mugs,DFT3,args.max_atoms),
                                   build_items(qval,DFT3,args.max_atoms)), DFT3),
          'qmugs_rdkit':(lambda:(build_items(random_mugs,[],args.max_atoms,rdkit=True),
                                 build_items(qval,[],args.max_atoms,rdkit=True)),
                         ['MolWt','TPSA','LogP','HBD']),
          'qmugs_matched_rdkit':(lambda:(build_items(matched_mugs,[],args.max_atoms,rdkit=True),
                                         build_items(qval,[],args.max_atoms,rdkit=True)),
                                  ['MolWt','TPSA','LogP','HBD'])
        }
        for variant in VARIANTS:
            v5.set_seed(seed)
            if variant=='scratch':
                gnn=base.MolecularGNN(hidden=args.hidden,output=1)
                pre_info={}
            else:
                (qtrain,qvalid),targets=(sources[variant][0](),sources[variant][1])
                expected_val = args.qm9_val_n if variant == 'qm9_dft' else args.qmugs_val_n
                assert len(qtrain)==args.n_pretrain and len(qvalid)==expected_val
                encoder=base.MolecularGNN(hidden=args.hidden,output=len(targets))
                print(f'--- seed={seed}: {variant}, {len(qtrain)} train, {len(qvalid)} val ---',flush=True)
                pre_info=v5.pretrain(encoder,qtrain,qvalid,targets,args,f'{variant}:{seed}')
                for target in targets:
                    pre_rows.append({'seed':seed,'variant':variant,'n_pretrain':len(qtrain),'target':target,
                      'mae':pre_info['pretrain_val_mae'][target],
                      'rmse':pre_info['pretrain_val_rmse'][target],
                      'r2':pre_info['pretrain_val_r2'][target],
                      'n_validation':len(qvalid),'best_epoch':pre_info['pretrain_best_epoch']})
                v5.set_seed(seed)
                gnn=base.MolecularGNN(hidden=args.hidden,output=1)
                gnn.embed.load_state_dict(encoder.embed.state_dict())
                gnn.layers.load_state_dict(encoder.layers.state_dict())
                del qtrain,qvalid,encoder
            v5.set_seed(seed+917)
            fit_info=v5.finetune(gnn,train,val,args,f'{variant}:{seed}')
            scores,probs=v5.evaluate(gnn,evaluation,args,fit_info['threshold'])
            rows.append({'seed':seed,'variant':variant,'n_pretrain':0 if variant=='scratch' else args.n_pretrain,
                         **pre_info,**fit_info,**scores})
            predictions[variant]=probs
            if args.save_checkpoints:
                (out/'checkpoints').mkdir(exist_ok=True)
                torch.save(gnn.state_dict(),out/'checkpoints'/f'{variant}_seed{seed}.pt')
            print(f'{variant} seed={seed} DEVELOPMENT metrics: {scores}',flush=True)
            del gnn
        predictions.to_csv(out/f'dev_predictions_seed{seed}.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'development_results.csv',index=False)
        pd.DataFrame(pre_rows).to_csv(out/'pretraining_metrics.csv',index=False)
        pd.DataFrame(audit).to_csv(out/'selection_audit.csv',index=False)
        pd.DataFrame(selection_rows).to_csv(out/'pretraining_selection.csv',index=False)
    df=pd.DataFrame(rows)
    summary=(df.groupby('variant').agg(runs=('seed','count'),
      mean_roc=('roc_auc','mean'),sd_roc=('roc_auc','std'),
      mean_pr=('pr_auc','mean'),sd_pr=('pr_auc','std'),
      mean_f1=('f1','mean'),sd_f1=('f1','std'),
      mean_mcc=('mcc','mean'),sd_mcc=('mcc','std')).reset_index())
    summary.to_csv(out/'summary.csv',index=False)
    scratch=df[df.variant=='scratch'][['seed','roc_auc','pr_auc','f1','mcc']].rename(
        columns={c:'scratch_'+c for c in ['roc_auc','pr_auc','f1','mcc']})
    paired=df[df.variant!='scratch'].merge(scratch,on='seed',validate='many_to_one')
    for metric in ('roc_auc','pr_auc','f1','mcc'):
        paired[f'delta_{metric}_vs_scratch']=paired[metric]-paired[f'scratch_{metric}']
    paired.to_csv(out/'paired_deltas.csv',index=False)
    controls=paired_control_deltas(df)
    controls.to_csv(out/'paired_controls.csv',index=False)
    controls.groupby('contrast',sort=False)[['delta_roc_auc','delta_pr_auc','delta_f1','delta_mcc']].agg(['mean','std']).to_csv(out/'paired_controls_summary.csv')
    hashes=pd.DataFrame(audit).pivot(index='seed',columns='variant',values='selection_sha256')
    if not (hashes.qmugs_matched == hashes.qmugs_matched_rdkit).all():
        raise RuntimeError('Matched DFT and matched RDKit molecular identity sets differ')
    if not (hashes.qmugs_random == hashes.qmugs_rdkit).all():
        raise RuntimeError('Random DFT and random RDKit molecular identity sets differ')
    manifest={'study':'QM9/QMugs transfer with matched DFT versus matched RDKit control',
       'status':'DEVELOPMENT ONLY, NOT FINAL TEST',
       'frozen_v03_test_excluded':heldout_n,
       'frozen_prior_split_sha256':hashlib.sha256(Path(args.prior_split).read_bytes()).hexdigest(),
       'qmugs_pool_rows':int(len(pool)),
       'qmugs_conformer_policy':'One conf_00 per ChEMBL identifier; NOT minimum-energy conformer',
       'qmugs_selection':'random vs nearest descriptor match to each seed INNER TRAIN only, without labels',
       'identity_control':'Matched RDKit and matched DFT use exactly same QMugs InChIKeys and pretraining validation IDs; random DFT and random RDKit use same IDs; SHA256 audit included',
       'shared_dft_targets':DFT3,
       'non_equivalent_dft_levels':'QM9 B3LYP and QMugs omegaB97X-D/def2-SVP; source comparison confounds method and chemical coverage',
       'limitations':['QM9-vs-QMugs difference cannot be uniquely attributed to domain shift',
           '2D GNN has no 3D conformer coordinates despite 3D-dependent QM labels',
           'Pretraining validation sets differ by source',
           'Overlapping development folds across seeds, no independent significance',
           'DFT-versus-RDKit compares DIFFERENT pretraining targets with differing learning difficulty, not a pure causal DFT effect',
           'Threshold-based F1/MCC can be unstable; thresholds chosen on inner validation only'],
       'num_seeds':len(args.seeds),'n_pretrain':args.n_pretrain,
       'task':args.task,'qmugs_sample_seed':args.qmugs_sample_seed,
       'qm9_sample_seed':args.qm9_sample_seed}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n=== DEVELOPMENT-ONLY SUMMARY ===\n'+summary.round(4).to_string(index=False),flush=True)
    print('SAVED:',out.resolve(),flush=True)


if __name__=='__main__':
    main()
