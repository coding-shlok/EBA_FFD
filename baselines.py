"""
baselines.py
============
Comparison baselines  (contribution #3).

v1 compared only against a centralized version of its own CNN-BiLSTM, which
answers "does federating cost accuracy?" but not "is this architecture worth
using at all?" A reviewer's first question about any deep fraud detector is
whether it beats gradient-boosted trees, which remain the strongest general
method on tabular data and are what banks actually deploy.

Baselines implemented
─────────────────────
  xgboost         Centralized XGBoost on pooled client data. The one to beat.
  logreg          Centralized L2 logistic regression. Sanity floor — if the deep
                  model cannot clear this, nothing else in the paper matters.
  centralized_nn  The proposed architecture (CNN, no BiLSTM — see config.py)
                  trained centrally. Isolates the cost of federating from the
                  cost of the architecture.
  fedprox         FedProx (Li et al., 2020): FedAvg plus a proximal term
                  mu/2 ||w - w_global||^2. The standard federated baseline for
                  non-IID data, and the fair comparison for adaptive aggregation.
  local_only      Each bank trains alone (XGBoost), metrics averaged. Quantifies
                  what collaboration actually buys — the motivating number for
                  the whole paper.

Fairness rules applied to every baseline
────────────────────────────────────────
  - identical train / validation / test splits
  - identical resampled client data
  - decision threshold calibrated on validation, never on test (evaluation.py)
  - reported over the same seeds as the proposed method

IMPORTANT INTERPRETIVE NOTE for the paper
─────────────────────────────────────────
xgboost, logreg, centralized_nn and local_only carry NO privacy guarantee: they
either pool raw data across banks or ignore privacy entirely. The proposed
system operates at a finite epsilon. So these are NOT like-for-like competitors
— they are utility upper bounds that quantify the price of privacy. Presenting
a DP method as "beating" a non-private one without that caveat would be
misleading. fedprox is the only baseline that can be run at matched epsilon,
and it is therefore the honest head-to-head comparison.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from evaluation import evaluate_probs
from config import get_config
# Re-exported for backward compatibility — these now live in tree_baselines.py
# so they can be imported WITHOUT pulling in torch (config.py imports torch
# unconditionally). Calling them directly here, in-process, still segfaults if
# torch has already been imported in this process; use run_all_baselines(),
# which routes them through a torch-free subprocess (see below).
from tree_baselines import run_xgboost, run_local_only


def run_logreg(resampled_clients, X_val, y_val, X_test, y_test, seed=42):
    """Centralized L2 logistic regression (no privacy). Sanity floor."""
    from sklearn.linear_model import LogisticRegression

    X = np.vstack([c["X"] for c in resampled_clients])
    y = np.hstack([c["y"] for c in resampled_clients])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=seed, n_jobs=1)
    clf.fit(X, y)

    return evaluate_probs(y_val,  clf.predict_proba(X_val)[:, 1],
                          y_test, clf.predict_proba(X_test)[:, 1])


# ── Torch-free subprocess for the XGBoost-based baselines ────────────────────
def _run_tree_baselines_subprocess(resampled_clients, X_val, y_val, X_test,
                                   y_test, seed, names):
    """
    Run xgboost / local_only in a SEPARATE process.

    PyTorch and XGBoost each bundle their own OpenMP runtime on macOS; by the
    time run_all_experiments.py reaches the baselines stage, torch has already
    been imported (stage_main runs first), and fitting an XGBClassifier in
    that same process reliably segfaults — verified empirically, no training
    required to reproduce it. tree_baselines_subprocess.py imports only
    tree_baselines.py, which never touches torch, so this subprocess is safe
    regardless of what the parent process has already loaded.
    """
    script = os.path.join(os.path.dirname(__file__),
                          "tree_baselines_subprocess.py")

    with tempfile.TemporaryDirectory() as d:
        in_path  = os.path.join(d, "in.npz")
        out_path = os.path.join(d, "out.json")

        arrays = {"X_val": X_val, "y_val": y_val,
                  "X_test": X_test, "y_test": y_test,
                  "n_clients": len(resampled_clients)}
        for i, c in enumerate(resampled_clients):
            arrays[f"client_{i}_X"] = c["X"]
            arrays[f"client_{i}_y"] = c["y"]
        np.savez(in_path, **arrays)

        r = subprocess.run([sys.executable, script, in_path, out_path, str(seed)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[baselines] !! tree-baseline subprocess failed "
                  f"(code {r.returncode}):\n{r.stderr[-2000:]}")
            return {name: {} for name in names}

        with open(out_path) as f:
            result = json.load(f)
        return {name: result.get(name, {}) for name in names}


# ── Federated baselines ───────────────────────────────────────────────────────
def run_fedprox(base_config, data, mu=0.01, seed=42):
    """
    FedProx at the SAME privacy budget as the proposed method — the one truly
    like-for-like comparison in this table.
    """
    from training_loop import run_federated_training

    cfg = get_config(**{**base_config})
    cfg.update({
        "aggregation": "fedprox",
        "fedprox_mu":  mu,
        "seed":        seed,
        "visualize":   False,
    })
    results, *_ = run_federated_training(cfg, X_cache=data)
    return results["best_metrics"]


def run_vanilla_fedavg(base_config, data, seed=42):
    """
    Classical FedAvg — the direct control for Adaptive FedAvg. Identical in
    every respect except that trust scores are held at 1.0.
    """
    from training_loop import run_federated_training

    cfg = get_config(**{**base_config})
    cfg.update({"aggregation": "fedavg", "seed": seed, "visualize": False})
    results, *_ = run_federated_training(cfg, X_cache=data)
    return results["best_metrics"]


def run_centralized_nn(base_config, data, resampled, seed=42):
    """The proposed architecture trained centrally (no federation, no DP)."""
    from training_loop import train_centralized_baseline
    from evaluation import score_at_threshold

    cfg = get_config(**{**base_config})
    cfg.update({"seed": seed, "dp_mode": "none"})

    return train_centralized_baseline(
        resampled, data["X_test"], data["y_test"],
        data["X_test"].shape[1], cfg["device"], config=cfg, epochs=5,
        X_val=data["X_val"], y_val=data["y_val"])


# ── Driver ────────────────────────────────────────────────────────────────────
BASELINE_REGISTRY = {
    "xgboost":        "Centralized XGBoost (no privacy)",
    "logreg":         "Centralized Logistic Regression (no privacy)",
    "local_only":     "Local-only XGBoost, averaged (no collaboration)",
    "centralized_nn": "Centralized CNN (no privacy)",
    "fedavg":         "Vanilla FedAvg + DP (matched epsilon)",
    "fedprox":        "FedProx + DP (matched epsilon)",
}


def run_all_baselines(base_config, data, resampled, seed=42,
                      include=None, verbose=True):
    """
    Run the requested baselines on a single seed.

    Returns {baseline_name: metrics_dict}.
    """
    include = include or list(BASELINE_REGISTRY.keys())
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    out = {}

    # xgboost and local_only both fit an XGBClassifier, which segfaults if run
    # in a process where torch is already imported (see
    # _run_tree_baselines_subprocess). Route both through one subprocess call
    # up front, then handle everything else in-process as before.
    tree_names = [n for n in ("xgboost", "local_only") if n in include]
    if tree_names:
        if verbose:
            for name in tree_names:
                print(f"\n[baselines] === {name}: {BASELINE_REGISTRY[name]} "
                      f"(torch-free subprocess) ===")
        tree_out = _run_tree_baselines_subprocess(
            resampled, X_val, y_val, X_test, y_test, seed, tree_names)
        for name in tree_names:
            out[name] = tree_out.get(name, {})
            if verbose and out[name]:
                m = out[name]
                print(f"[baselines] {name}: f1={m.get('f1', 0):.4f} "
                      f"recall={m.get('recall', 0):.4f} "
                      f"pr_auc={m.get('pr_auc', 0):.4f}")

    for name in include:
        if name in tree_names:
            continue
        if verbose:
            print(f"\n[baselines] === {name}: {BASELINE_REGISTRY[name]} ===")
        try:
            if name == "logreg":
                m = run_logreg(resampled, X_val, y_val, X_test, y_test, seed)
            elif name == "centralized_nn":
                m = run_centralized_nn(base_config, data, resampled, seed)
            elif name == "fedavg":
                m = run_vanilla_fedavg(base_config, data, seed)
            elif name == "fedprox":
                m = run_fedprox(base_config, data,
                                mu=base_config.get("fedprox_mu", 0.01), seed=seed)
            else:
                continue

            out[name] = m
            if verbose and m:
                print(f"[baselines] {name}: f1={m.get('f1', 0):.4f} "
                      f"recall={m.get('recall', 0):.4f} "
                      f"pr_auc={m.get('pr_auc', 0):.4f}")
        except Exception as e:
            print(f"[baselines] !! {name} failed: {type(e).__name__}: {e}")
            out[name] = {}

    return out


if __name__ == "__main__":
    from data_loader import load_federated_data
    from preprocessing import preprocess_clients

    cfg  = get_config(num_clients=3, subsample_frac=0.05, verbose=False)
    data = load_federated_data(cfg)
    resampled, _ = preprocess_clients(data["clients"], cfg, visualize=False)

    res = run_all_baselines(cfg, data, resampled, seed=42,
                            include=["xgboost", "logreg", "local_only"])
    for k, v in res.items():
        print(f"{k:16s} f1={v.get('f1', 0):.4f}  recall={v.get('recall', 0):.4f}")
        
        