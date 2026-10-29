# Data inputs (not included in this source archive)

The full local experimental project uses files stored on the original machine.
The downloaded datasets and local experiment outputs **have not been supplied to the archive builder**.
The ~2.03 GB `data/qmugs/summary.csv`, ~32k QMugs extraction pool, and historical split
`results/gnn_v03_5k/tox_split.csv` are therefore **not** in this GitHub package.

First install requirements and run `python restore_data.py` (downloads public QM9 and
Tox21, generates `data/tox21_clean.csv.gz`, if connected). Obtain QMugs Summary from:
https://huggingface.co/datasets/JuIm/QMugs_Summary

Save the QMugs source as `data/qmugs/summary.csv` and generate its sample pool with
`python extract_qmugs_v06.py --source data/qmugs/summary.csv \
 --sample-rate 0.05 --chunksize 20000 \
 --out data/qmugs/qmugs_v06_pool.csv.gz \
 --audit-out data/qmugs/qmugs_v06_audit.json`

To reproduce v0.8/v0.9 **exactly**, additionally copy your previously saved local
historical split artifact to `results/gnn_v03_5k/tox_split.csv`, and retain its
checksums, as well as existing v0.8 manifests. **Do not substitute a newly generated
split and claim exact reproduction**. Ensure data licensing and provenance.

The experiment scripts intentionally do not overwrite recorded outputs.
