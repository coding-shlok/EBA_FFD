"""
evaluation.py
=============
Shared scoring utilities, so every model in the paper — federated, XGBoost,
logistic regression, every ablation — is measured by exactly the same procedure.

The single most important guarantee here: the decision threshold is always
chosen on VALIDATION data and then applied unchanged to test data. Applying a
different rule to different models (e.g. a fixed 0.5 for the baselines but a
tuned threshold for the proposed method) would manufacture an advantage and is
the kind of thing reviewers check for.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def best_threshold(y_val, probs_val, objective="f1"):
    """
    Select a decision threshold on validation data.

    objective:
      "f1"        maximise validation F1
      "recall@p"  maximise recall subject to precision >= p
    """
    if len(np.unique(y_val)) < 2:
        return 0.5

    candidates = np.unique(np.quantile(probs_val, np.linspace(0.5, 0.99999, 400)))
    best_thr, best_score = 0.5, -1.0

    for thr in candidates:
        preds = (probs_val >= thr).astype(int)
        if preds.sum() == 0:
            continue
        if objective.startswith("recall@"):
            p_min = float(objective.split("@")[1])
            prec  = precision_score(y_val, preds, zero_division=0)
            score = recall_score(y_val, preds, zero_division=0) if prec >= p_min else -1.0
        else:
            score = f1_score(y_val, preds, zero_division=0)
        if score > best_score:
            best_score, best_thr = score, float(thr)

    return best_thr


def score_at_threshold(y_true, probs, threshold):
    """Full metric set at a fixed threshold."""
    preds = (np.asarray(probs) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    metrics = {
        "threshold": float(threshold),
        "accuracy":  accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall":    recall_score(y_true, preds, zero_division=0),
        "f1":        f1_score(y_true, preds, zero_division=0),
        "auc":       roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc":    average_precision_score(y_true, probs)
                     if len(np.unique(y_true)) > 1 else 0.0,
    }

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    metrics.update({
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        # Operationally meaningful for a fraud team: how many alerts must an
        # analyst review per genuine fraud caught?
        "alerts_per_fraud": float((tp + fp) / tp) if tp > 0 else float("inf"),
    })
    return metrics


def evaluate_probs(y_val, probs_val, y_test, probs_test, objective="f1"):
    """Calibrate on validation, then score on test. The standard path."""
    thr = best_threshold(y_val, probs_val, objective=objective)
    return score_at_threshold(y_test, probs_test, thr)


METRIC_KEYS = ["precision", "recall", "f1", "auc", "pr_auc"]
