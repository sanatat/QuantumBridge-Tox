# QuantumBridge-Tox — concise research summary

**Project:** Domain-aware quantum pretraining for molecular AhR bioactivity prediction (Python/PyTorch, RDKit; CPU-only)

I built QuantumBridge-Tox to investigate whether learning electronic properties from quantum-chemical datasets can improve molecular graph neural network (GNN) predictions of Tox21 NR-AhR activity. After identifying substantial chemical-domain mismatch between QM9 and Tox21 (only 10.23% of unique Tox21 compounds met a simple QM9-like size/element criterion), I created a reproducible QMugs-based pretraining pipeline with domain-matched and random molecular selection, RDKit-descriptor controls, and shuffled-DFT-label negative controls.

In a five-initialization fixed-scaffold development pilot, matched-QMugs DFT pretraining achieved mean PR-AUC 0.3704 versus 0.3432 for a scratch GNN. Across three independently shuffled target maps and five training seeds, genuine DFT targets exceeded shuffled controls in 14/15 correlated comparisons (mean paired PR-AUC delta +0.0282). The advantage over matched RDKit pretraining was small (+0.0047), and a fingerprint Random Forest remained stronger (PR-AUC 0.3981). Thus the project supports meaningful structure-linked pretraining relative to randomized supervision, without establishing quantum-specific superiority or external generalization. I am now preparing independent assay-cohort validation with strict chemical identity, assay provenance, and scaffold-overlap auditing.

**Status:** Exploratory development study; independent external validation not yet performed. Original historical test was inspected in an early pilot and must not be described as globally blind.
