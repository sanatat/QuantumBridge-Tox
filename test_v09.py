#!/usr/bin/env python3
"""Fast deterministic controls tests; optional tiny CPU training integration test."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gnn_quantumbridge as base
import gnn_quantumbridge_v09 as v9


def test_shuffle():
    items = []
    for i in range(51):
        graph = ('graph-id', i)
        y = torch.tensor([float(i), float(i*i % 37), float(i + 100)], dtype=torch.float32)
        items.append((graph, y))
    a, pa = v9.shuffle_target_rows(items, 128)
    b, pb = v9.shuffle_target_rows(items, 128)
    c, pc = v9.shuffle_target_rows(items, 129)
    assert np.array_equal(pa, pb)
    assert not np.array_equal(pa, pc)
    assert all(x[0] is y[0] for x, y in zip(a, items))
    assert all(torch.equal(x[1], y[1]) for x, y in zip(a, b))
    assert sum(int(i != j) for i, j in enumerate(pa)) > 40
    real = np.stack([t.numpy() for _, t in items])
    fake = np.stack([t.numpy() for _, t in a])
    assert np.array_equal(np.sort(real,axis=0),np.sort(fake,axis=0))
    assert np.array_equal(real[:,2] - real[:,0],fake[:,2] - fake[:,0])
    assert not np.array_equal(real, fake)
    keys = [f'MOCK-{i}' for i in range(51)]
    rec = v9.assignment_table(keys, pa, 'train')
    assert rec.graph_inchikey.nunique() == 51
    assert rec.label_source_inchikey.nunique() == 51
    print('PASS: joint-row target shuffle; unchanged graphs, distribution, row correlation, fixed null seed')


def test_training():
    # Tiny actual CPU learning step; no Tox21/QMugs data needed.
    rng = np.random.default_rng(891)
    qtrain = []
    qval = []
    smiles = ['C','CC','CCC','CCCC','CCO','CCN','CCF','CO','CN','CCCO','CCCN','CCOC','CCOCN',
              'c1ccccc1','c1ccncc1','CCCl','CCBr','COC','CC(=O)O','O=C=O']
    for ix in range(120):
        mol = base.to_graph(smiles[ix % len(smiles)], max_atoms=128)
        y = torch.tensor(rng.normal(size=3).astype(np.float32))
        (qtrain if ix < 96 else qval).append((mol,y))
    fake_train, _ = v9.shuffle_target_rows(qtrain, 123)
    fake_val, _ = v9.shuffle_target_rows(qval, 124)
    drows, items = [], []
    for i in range(94):
        smi = smiles[(i * 3) % len(smiles)]
        y = int(i % 7 == 0)
        g = base.to_graph(smi, max_atoms=128)
        drows.append({'inchikey':f'DEV-{i}','NR-AhR':y})
        items.append((g,torch.tensor([float(y)])))
    dev = pd.DataFrame(drows)
    settings = argparse.Namespace(hidden=12, pretrain_epochs=2, finetune_epochs=2,
                                  patience=2, batch=16, lr=.001, task='NR-AhR',
                                  save_checkpoints=False, threads=2,
                                  out=str(Path(tempfile.mkdtemp())), max_atoms=128)
    torch.set_num_threads(2)
    train_idx=np.arange(0,60); val_idx=np.arange(60,75); eval_idx=np.arange(75,94)
    assert dev.iloc[val_idx]['NR-AhR'].sum() > 0
    result, pre, pred = v9.run_seed(42, settings, dev, items, train_idx, val_idx, eval_idx,
                                   fake_train, fake_val, qtrain, qval)
    assert len(result) == 1 and len(pre) == 3 and len(pred) == len(eval_idx)
    assert result.iloc[0].variant == v9.CONTROL
    assert 0 <= result.iloc[0].pr_auc <= 1
    assert set(pre.target) == set(v9.LABELS)
    assert pred.label.sum() > 0
    print('PASS: full CPU shuffled pretraining and toxicity finetuning with true-label diagnostic')


if __name__ == '__main__':
    test_shuffle()
    test_training()
