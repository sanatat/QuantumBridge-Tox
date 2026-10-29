# QuantumBridge-Tox: External AhR validation protocol (candidate-screening stage)

**Version/date:** 2026-10-29. **Status: PLANNED, NOT EXECUTED.**

## Outcome / dataset definition

Primary model comparison should be **QMugs matched DFT vs QMugs matched RDKit**, with Random Forest and scratch GNN reported as secondary/practical baselines. A second predeclared comparison is QMugs matched DFT vs scratch; external PR-AUC (average precision) is primary, ROC-AUC secondary. Do not choose a new model or numerical threshold based on external labels.

A suitable external candidate contains independently tested **AhR agonist-like activation** measurements with an assay protocol and chemical structure. Do not combine agonist, antagonist, binding-only and cytotoxicity observations as though they were one endpoint. Do not infer inactive from missing measurements. Before converting raw assay data into binary labels, write down rules for positive, negative, inconclusive, control-interference/cytotoxicity, replicate disagreement and censored results; retain assay IDs, source DOI, organism, technology, activity direction and release/version. Avoid adding new data once the external evaluation is viewed.

**EPA ToxCast** and its invitrodb source are candidate discovery resources, not presumptively independent validation datasets: the ToxCast umbrella includes Tox21 program data. **ChEMBL** is another assay discovery resource but includes heterogeneous measurements. Either source needs manual assay selection and provenance audit. The supported source CSV is a **manually curated table**, not a raw ChEMBL/ToxCast activity export.

## Exact minimum CSV schema

```
smiles,label,assay_id,source,source_document,activity_direction,species
CCO,0,EXAMPLE_ASSAY_X,EXAMPLE_SOURCE,EXAMPLE_DOI,agonist,Homo sapiens
```

The demonstration row is NOT a real external observation and must not be used for scoring. `label` must already be curated as 0 or 1 from a specified experimentally measured assay. An additional `confidence` or `replicates` column is recommended. Use one homogeneous assay for the primary external cohort.

## Pre-registered screening and exclusions

1. Record source/version/download timestamp, raw SHA256, endpoint/assay semantics, label rule and number of compounds BEFORE examining predictions.
2. Reject invalid SMILES, mixtures/salts for which no controlled normalization is defined, unsupported graph size (current model max 128 atoms), missing/conflicting labels and duplicate molecular identities; do not silently relabel.
3. Generate RDKit canonical SMILES, InChIKey and Bemis-Murcko scaffold with a logged RDKit version. Reject exact molecular identity overlap with *all* cleaned Tox21 compounds (including training/dev/prior 1,219-compound test); report scaffold overlaps separately; also report overlap with the **pretraining QMugs pool** and exclude those identities from the primary strictly unseen-chemical analysis.
4. Treat connected source histories carefully: ToxCast includes Tox21; source-specific duplicates or shared assays should be removed even after InChIKey screening.
5. Require at least 20 positive and 20 negative retained compounds before calculating a meaningful primary PR-AUC; if fewer, report descriptive-only counts, not a claimed validation.
6. Run all five saved matched DFT, matched RDKit, and scratch checkpoints without tuning. Compare per-seed paired predictions on identical molecules. Rank by PR-AUC and report prevalence baseline.
7. If including Random Forest, reconstruct using the **fixed 3,609 Tox21 inner-training compounds only** and original RF hyperparameters/seed. Verify its development predictions against `predictions_seed*.csv` first. Refitting on more Tox21 data is a *new model* and not a comparable frozen-score baseline.
8. Report molecule-level and scaffold-level coverage, per-assay metrics, sample count/positives/prevalence, per-initialization scores, paired differences and model-version hashes. Use scaffold-cluster bootstrap on the *external chemical cohort* for uncertainty; treat all seeds as repeated predictions on the same specimens, not independent datasets.
9. Publish an attrition table for candidate input -> valid structures -> nonduplicated -> non-Tox21 overlap -> non-QMugs overlap -> valid class balance -> scored. Do NOT interpret inability to obtain a valid candidate as negative scientific evidence.

## Execution (once a real assay is curated)

```
cd ~/Downloads/QuantumBridge-Tox
source .venv/bin/activate
python research_pack/scripts/external_cohort_audit.py \
  --input data/external/curated_ahr.csv \
  --tox data/tox21_clean.csv.gz \
  --qmugs-pool data/qmugs/qmugs_v06_pool.csv.gz \
  --out results/external_preflight

# Review the audit JSON, assay provenance and external cohort manually.
# If approved, score the frozen GNN checkpoints:
python research_pack/scripts/score_external_checkpoints.py \
  --input results/external_preflight/curated_nonoverlap.csv \
  --v08 results/gnn_v08_fixed_split_5k \
  --approve-assay-id 'YOUR_VERIFIED_ASSAY_ID' \
  --out results/external_scores
```

The second script operates **only on the audited external CSV and existing trained checkpoints**. It does not select a new threshold; F1/MCC are intentionally omitted from primary external evaluation. Its metrics are only as meaningful as the externally curated assay label definitions.

## Model-selection freeze before a true final external run

- PR-AUC primary; ROC-AUC secondary; no tuning of epochs/architecture/compounds/seed selection after reviewing the external dataset.
- Predeclare models: matched DFT and matched RDKit as primary paired comparison; scratch and RF as secondary baselines.
- Use equal five initialization seeds and report paired per-seed external PR-AUC differences, not only grand means.
- Include external dataset ID, positive-case count, label version, duplicates/overlap exclusions and quality controls in any claim.
- Since v0.3 test was inspected historically, do NOT call it globally blind; prioritize a truly independent external source.

**Verified source starting points:** U.S. EPA ToxCast invitrodb v4.3 official resource https://www.epa.gov/comptox-tools/exploring-toxcast-data ; ChEMBL data API https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services .
