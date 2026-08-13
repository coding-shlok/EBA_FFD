"""
tree_baselines.py
==================
XGBoost-based baselines (xgboost, local_only), split out from baselines.py so
they can run in a process that NEVER imports torch.

Why this module exists
───────────────────────
PyTorch and XGBoost each bundle their own OpenMP runtime on macOS. Importing
both in the same process reliably segfaults — verified empirically: even
`import torch` followed by an unrelated `XGBClassifier().fit(...)` crashes,
with no training involved. config.py imports torch unconditionally (for CUDA
device selection), so anything that does `from config import ...` — including
baselines.py — pulls torch in transitively. This module imports nothing that
touches torch (numpy, sklearn-adjacent evaluation.py, xgboost only), so
tree_baselines_subprocess.py can import from here and run XGBoost safely in a
subprocess that a torch-using parent process spawns.
"""
import numpy as np

from evaluation import evaluate_probs


def run_xgboost(resampled_clients, X_val, y_val, X_test, y_test, seed=42):
    """Centralized XGBoost on pooled client data (no privacy)."""
    from xgboost import XGBClassifier

    X = np.vstack([c["X"] for c in resampled_clients])
    y = np.hstack([c["y"] for c in resampled_clients])

    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))

    clf = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = neg / pos,
        eval_metric      = "aucpr",
        random_state     = seed,
        n_jobs           =  1,
        tree_method      = "hist",
    )
    clf.fit(X, y)

    return evaluate_probs(y_val,  clf.predict_proba(X_val)[:, 1],
                          y_test, clf.predict_proba(X_test)[:, 1])


def run_local_only(resampled_clients, X_val, y_val, X_test, y_test, seed=42):
    """
    Each bank trains an XGBoost model alone on its own data; metrics are averaged
    across banks. This is the "no collaboration" counterfactual and is the number
    that justifies federated learning existing at all.
    """
    from xgboost import XGBClassifier

    per_client = []
    for c in resampled_clients:
        X, y = c["X"], c["y"]
        if len(np.unique(y)) < 2:
            continue
        pos = max(1, int((y == 1).sum()))
        neg = max(1, int((y == 0).sum()))
        clf = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                            scale_pos_weight=neg / pos, eval_metric="aucpr",
                            random_state=seed, n_jobs=1, tree_method="hist")
        clf.fit(X, y)
        per_client.append(evaluate_probs(
            y_val,  clf.predict_proba(X_val)[:, 1],
            y_test, clf.predict_proba(X_test)[:, 1]))

    if not per_client:
        return {}

    keys = ["precision", "recall", "f1", "auc", "pr_auc"]
    out = {k: float(np.mean([m[k] for m in per_client])) for k in keys}
    out["n_clients_averaged"] = len(per_client)
    return out
