#!/usr/bin/env python3
"""Fast local unit checks for v0.5 metric logic; no datasets or training needed."""
import numpy as np
from sklearn.metrics import r2_score
from gnn_quantumbridge_v05 import classification_at_threshold, choose_validation_threshold

# Four outcomes at cutoff 0.5: one TP, one TN, one FP, one FN.
y = np.array([1, 0, 1, 0])
p = np.array([0.9, 0.6, 0.4, 0.1])
r = classification_at_threshold(y, p, 0.5)
assert (r['tp'], r['tn'], r['fp'], r['fn']) == (1, 1, 1, 1)
assert abs(r['f1'] - 0.5) < 1e-10 and abs(r['mcc']) < 1e-10

# A perfect classifier admits a validation-selected cutoff with MCC/F1 == 1.
y = np.array([1, 1, 0, 0])
p = np.array([0.9, 0.8, 0.2, 0.1])
t, score = choose_validation_threshold(y, p)
assert 0.2 < t <= 0.8 and score['mcc'] == 1.0 and score['f1'] == 1.0

# Regression R² is 1 for identical predictions; not the QM9 r2 field.
z = np.array([1., 2., 3., 4.])
assert abs(r2_score(z, z)-1.0) < 1e-12
print('PASS: threshold selection, F1, MCC, and R² metric checks')
