"""
training_loop.py
================
Orchestrator for a single EBA-FFD federated run.

Pipeline:
  1. Load & partition data into non-IID federated clients
  2. Balance each client's data locally (KMeans-SMOTE -> SMOTE fallback + ENN)
  3. Calibrate a per-bank DP noise multiplier for a common epsilon target
  4. Federated rounds:
       a. (optional) inject concept drift into selected clients
       b. server broadcasts global weights
       c. clients train locally under DP-SGD, return weights + mean loss
       d. server runs Page-Hinkley on each client's loss -> trust scores
       e. server aggregates with size x trust weighting  (Adaptive FedAvg)
       f. server evaluates the global model on the held-out test set
  5. Save metrics, curves, trust dynamics

This module deliberately returns a rich results dict rather than printing
everything, so that ablations.py / scaling.py / baselines.py can call it in a
loop and aggregate across seeds without re-parsing stdout.
"""

import json
import os
import time

import numpy as np
import torch

from config        import get_config
from data_loader   import load_federated_data
from preprocessing import preprocess_clients
from model         import build_model, build_loss, count_parameters
from client        import build_clients
from server        import FederatedServer
from drift         import inject_drift

OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG = get_config()          # backwards compatibility with v1 imports


def set_seed(seed):
    """Seed every RNG that can affect a run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_federated_training(config=None, X_cache=None, verbose=None):
    """
    Execute one complete federated run.

    X_cache: optional pre-loaded (client_data, X_test, y_test, feature_names,
             scaler, meta) tuple. Loading and partitioning the 284k-row CSV
             takes longer than a short training run, so the experiment drivers
             pass a cached partition when sweeping seeds or configurations that
             share the same data split.

    Returns a results dict.
    """
    config = config or get_config()
    if verbose is not None:
        config["verbose"] = verbose
    verbose = config.get("verbose", True)

    set_seed(config["seed"])
    device = config["device"]
    t0 = time.time()

    if verbose:
        print(f"\n{'='*66}")
        print(f"  EBA-FFD v2 | agg={config['aggregation']} | "
              f"dp={config['dp_mode']} | clients={config['num_clients']} | "
              f"seed={config['seed']}")
        print(f"{'='*66}")

    # ── Step 1: data ──────────────────────────────────────────────────────────
    data = X_cache if X_cache is not None else load_federated_data(config)
    client_data   = data["clients"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_val,  y_val  = data["X_val"],  data["y_val"]
    feature_names, meta = data["feature_names"], data["meta"]
    input_dim = X_test.shape[1]

    # ── Step 2: balancing ─────────────────────────────────────────────────────
    resampled, balance_info = preprocess_clients(
        client_data, config, visualize=config.get("visualize", False))

    # ── Step 3: server + clients + privacy calibration ────────────────────────
    server = FederatedServer(input_dim=input_dim, device=device, config=config,
                             num_clients=len(resampled))
    clients, privacy_plan = build_clients(resampled, input_dim=input_dim,
                                          device=device, config=config)

    # ── Step 4: federated rounds ──────────────────────────────────────────────
    drift_applied_round = None
    for rnd in range(1, config["num_rounds"] + 1):
        if verbose:
            print(f"\n{'─'*50}\n  Round {rnd}/{config['num_rounds']}\n{'─'*50}")

        # [a] drift injection (evaluation harness for contribution #1)
        if config.get("drift_enabled") and rnd == config.get("drift_round"):
            for cid in config.get("drift_clients", []):
                if cid < len(clients):
                    inject_drift(clients[cid],
                                 mode  = config.get("drift_mode", "covariate_shift"),
                                 level = config.get("drift_level", 0.5),
                                 seed  = config["seed"] + cid)
            drift_applied_round = rnd
            if verbose:
                print(f"[training_loop] ⚡ drift injected into clients "
                      f"{config.get('drift_clients')} "
                      f"(mode={config.get('drift_mode')})")

        # [b] broadcast
        global_weights = server.get_global_weights()

        # [c] local training
        weights_list, sizes, cmetrics = [], [], []
        for c in clients:
            c.receive_global_weights(global_weights)
            w, m = c.local_train()
            m["client_id"] = c.client_id
            weights_list.append(w)
            sizes.append(c.n_samples)
            cmetrics.append(m)

        # [d] drift detection -> trust
        trusts, _ = server.update_trust(cmetrics, round_num=rnd)

        # [e] adaptive aggregation
        aggregated, p = server.aggregate(weights_list, sizes, trusts=trusts)
        server.update_global_model(aggregated)

        # [f] global evaluation (threshold recalibrated on validation first)
        server.evaluate_global(X_test, y_test, round_num=rnd,
                               X_val=X_val, y_val=y_val)

    elapsed = time.time() - t0

    # ── Step 5: outputs ───────────────────────────────────────────────────────
    if config.get("visualize", False):
        server.plot_training_curves()
        server.plot_confusion_matrix(X_test, y_test)
        server.plot_trust_dynamics(drift_round=drift_applied_round)
        server.save_final_model()

    final = server.round_metrics[-1] if server.round_metrics else {}
    # Select the reported "best" round by VALIDATION PR-AUC, never test
    # PR-AUC — the test set must stay untouched by any decision that affects
    # what gets reported. PR-AUC, not F1, because F1 is scored at a threshold
    # that's re-optimised against validation every round, which lets near-peak
    # val F1 be hit most rounds — so it ties across rounds and max() silently
    # falls back to the first (earliest) tied round. PR-AUC has no threshold
    # to re-optimise against, so it actually discriminates which round is
    # best. Falls back to test PR-AUC only if a run has no validation set.
    def _select_key(m):
        return m["val_pr_auc"] if m.get("val_pr_auc") is not None else m["pr_auc"]
    best = max(server.round_metrics, key=_select_key) \
            if server.round_metrics else {}

    results = {
        "config":          _serialisable(config),
        "final_metrics":   final,
        "best_metrics":    best,
        "round_metrics":   server.round_metrics,
        "privacy_plan":    privacy_plan,
        "epsilon_spent":   [c.epsilon_spent for c in clients],
        "balancing_info":  balance_info,
        "partition_meta":  meta,
        "aggregation_weights": server.weight_history,
        "drift_records":   server.drift_records,
        "drift_round":     drift_applied_round,
        "client_sizes":    [c.n_samples for c in clients],
        "n_params":        count_parameters(server.global_model),
        "elapsed_sec":     elapsed,
        "threshold":       getattr(server, "threshold", 0.5),
    }

    if verbose:
        print(f"\n[training_loop] done in {elapsed:.1f}s | "
              f"best F1={best.get('f1', 0):.4f} recall={best.get('recall', 0):.4f}")

    return results, server, clients, data, resampled


# ── Centralized deep baseline (kept from v1) ──────────────────────────────────
def train_centralized_baseline(resampled_clients, X_test, y_test, input_dim,
                               device, config=None, epochs=5,
                               X_val=None, y_val=None):
    """
    Train the same CNN-BiLSTM centrally on pooled client data.

    NOTE: this violates the privacy premise — raw data from every bank is
    pooled. It exists solely as the utility upper bound the federated system is
    measured against.
    """
    from torch.utils.data import TensorDataset, DataLoader
    from evaluation import evaluate_probs, score_at_threshold

    config = config or get_config()
    X_all = np.vstack([c["X"] for c in resampled_clients])
    y_all = np.hstack([c["y"] for c in resampled_clients])

    loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                      torch.tensor(y_all, dtype=torch.long)),
        batch_size=512, shuffle=True)

    model     = build_model(input_dim, device, config)
    criterion = build_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b.to(device)), y_b.to(device))
            loss.backward()
            optimizer.step()

    model.eval()

    def _probs(X):
        with torch.no_grad():
            return torch.sigmoid(
                model(torch.tensor(X, dtype=torch.float32).to(device))
            ).cpu().numpy()

    probs_test = _probs(X_test)

    # Calibrate the decision threshold on validation data, exactly as the
    # federated system and every other baseline does. Scoring this model at a
    # fixed 0.5 while tuning the proposed method's threshold would be an unfair
    # comparison in our own favour.
    if X_val is not None:
        return evaluate_probs(y_val, _probs(X_val), y_test, probs_test)
    return score_at_threshold(y_test, probs_test, 0.5)


def _serialisable(obj):
    """Recursively coerce numpy/torch scalars into JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialisable(results), f, indent=2)
    return path


if __name__ == "__main__":
    cfg = get_config(num_rounds=3, num_clients=3, subsample_frac=0.05,
                     local_epochs=1, visualize=False)
    res, *_ = run_federated_training(cfg)
    print(json.dumps(_serialisable(res["final_metrics"]), indent=2))
