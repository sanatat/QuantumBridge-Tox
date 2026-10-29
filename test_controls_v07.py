#!/usr/bin/env python3
"""Small deterministic sanity tests; no training data or network needed."""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from gnn_quantumbridge_v07 import selected_qmugs, build_items, selection_digest, paired_control_deltas

smiles = ['CCO', 'CCN','CCCO','CCCCO','CCCN','CCCCN','c1ccccc1','Cc1ccccc1','Clc1ccccc1','Oc1ccccc1','CC(=O)O','CCC(=O)O','COC','COCC','CCOC','CCCOC']
rows=[]
for i,s in enumerate(smiles):
    m=Chem.MolFromSmiles(s)
    rows.append(dict(smiles=s,inchikey=Chem.MolToInchiKey(m),
        heavy_atoms=m.GetNumHeavyAtoms(),mw=Descriptors.MolWt(m),
        logp=Crippen.MolLogP(m),aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(m),
        halogen=int(any(a.GetSymbol() in ('Cl','Br','I') for a in m.GetAtoms())),
        mu=float(i+.2),homo=float(-.2+i*.001),lumo=float(.1+i*.002)))
pool=pd.DataFrame(rows)
dev=pd.DataFrame({'canonical_smiles':['CCO','CCN','CCCCO','Oc1ccccc1','Clc1ccccc1']})
random_set, matched=selected_qmugs(pool,pool.index[:2],42,dev,np.array([0,1,2,3]),6)
assert len(random_set)==6 and len(matched)==6
assert not (set(matched.index)&set(pool.index[:2]))
assert len(set(matched.inchikey))==6
assert selection_digest(matched)==selection_digest(matched.sample(frac=1,random_state=99))
dft=build_items(matched,['mu','homo','lumo'],128)
rdkit=build_items(matched,[],128,rdkit=True)
assert len(dft)==len(rdkit)==6
for a,b in zip(dft,rdkit):
    assert len(a[0][0])==len(b[0][0]) and a[1].numel()==3 and b[1].numel()==4
comparison=pd.DataFrame([
    {'seed':seed,'variant':variant,'roc_auc':.8+offset,'pr_auc':.3+offset,'f1':.4+offset,'mcc':.2+offset}
    for seed in [42,123] for variant,offset in {
    'qmugs_matched':.04,'qmugs_matched_rdkit':.02,'qmugs_random':.03,
    'qmugs_rdkit':.01,'scratch':0.0,'random_forest':.05}.items()
])
paired=paired_control_deltas(comparison)
assert len(paired)==12
r=paired[(paired.seed==42)&(paired.contrast=='matched DFT minus matched RDKit')].iloc[0]
assert abs(r.delta_pr_auc-.02)<1e-10
print('PASS: deterministic QMugs identity matching; DFT/RDKit same graphs; paired development comparisons')
