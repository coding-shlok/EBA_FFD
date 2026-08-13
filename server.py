"""
server.py
=========
Federated aggregation server — Adaptive FedAvg  (contribution #1).

Classical FedAvg (McMahan et al., 2017)
───────────────────────────────────────
    w_{t+1} = SUM_k (n_k / N) * w_k

Client k's influence depends only on its dataset size, and never changes. That
is the right estimator when every client's distribution is stationary, because
then each client's local empirical risk is an unbiased estimate of the same
global risk and sample-size weighting is the minimum-variance combination.

Adaptive FedAvg (this work)
───────────────────────────
Fraud distributions are not stationary. When a bank's local distribution
shifts, its update stops estimating the shared objective and starts pulling the
global model toward a different one. We therefore modulate each client's
aggregation weight by a trust score derived from a Page-Hinkley test on that
client's mean local loss (see drift.py):

    p_k = (n_k * trust_k) / SUM_j (n_j * trust_j)
    w_{t+1} = SUM_k p_k * w_k

Properties worth stating in the paper:
  - Reduces to FedAvg exactly when no drift is detected (all trust_k = 1), so
    the method is a strict generalisation and cannot hurt in the stationary
    case beyond detector false-alarm rate.
  - trust_k is lower-bounded by trust_min > 0, so a drifting bank is DAMPED,
    never silenced. This matters: after a genuine regime change, the drifting
    bank holds the only evidence about the new concept. Zeroing it out would
    make the federation permanently blind to the new fraud pattern.
  - Requires one scalar per client per round (mean local loss). The server sees
    no features and no gradients beyond what FedAvg already receives, so drift
    detection consumes NO additional privacy budget.
  - Costs O(K) time and O(K) memory per round.
"""

import copy
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, average_precision_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model, get_model_weights
from drift import ClientDriftMonitor

PLOT_DIR  = os.path.join(os.path.dirname(__file__), "outputs", "plots")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class FederatedServer:
    """
    Central aggregation server.

    Manages global model initialisation, (adaptive) aggregation, per-client
    drift monitoring, global evaluation, and training history.
    """

    def __init__(self, input_dim=30, device="cpu", config=None, num_clients=None):
        self.config      = config or {}
        self.device      = device
        self.input_dim   = input_dim
        self.aggregation = self.config.get("aggregation", "adaptive")

        self.global_model = build_model(input_dim, device, self.config)

        # One drift monitor per client
        n_clients = num_clients or self.config.get("num_clients", 4)
        self.monitors = {
            cid: ClientDriftMonitor(
                cid,
                detector_cfg = self.config.get("drift_detector", {}),
                beta         = self.config.get("trust_beta", 2.0),
                trust_min    = self.config.get("trust_min", 0.2),
                recovery     = self.config.get("trust_recovery", 0.3),
            )
            for cid in range(n_clients)
        }

        self.round_metrics    = []
        self.weight_history   = []   # aggregation weights per round
        self.drift_records    = []   # detector output per client per round
        self.best_f1          = 0.0
        self.best_weights     = None

        if self.config.get("verbose", True):
            print(f"[Server] Global model on {device} | "
                  f"aggregation = {self.aggregation}")

    # ── Weight distribution ───────────────────────────────────────────────────
    def get_global_weights(self):
        return get_model_weights(self.global_model)

    # ── Drift-aware trust scoring ─────────────────────────────────────────────
    def update_trust(self, client_metrics, round_num):
        """
        Feed each client's mean local loss into its Page-Hinkley detector and
        return the resulting trust vector.

        Under aggregation="fedavg"/"fedprox" the detectors still RUN and still
        log — so the paper can show what would have been detected — but the
        returned trust is all-ones, leaving those baselines untouched.
        """
        trusts, records = [], []
        for m in client_metrics:
            cid = m["client_id"]
            rec = self.monitors[cid].update(m["loss"], round_num=round_num)
            records.append(rec)
            trusts.append(rec["trust"])

        self.drift_records.extend(records)

        if self.aggregation != "adaptive":
            return [1.0] * len(trusts), records

        if any(r["detected"] for r in records) and self.config.get("verbose", True):
            flagged = [r["client_id"] for r in records if r["detected"]]
            print(f"[Server] ⚠ drift alarm at round {round_num}: clients {flagged}")

        return trusts, records

    # ── Aggregation ───────────────────────────────────────────────────────────
    def aggregate(self, client_weights_list, client_sizes, trusts=None):
        """
        Weighted parameter average.

            trusts=None or all 1.0  ->  classical FedAvg
            otherwise               ->  Adaptive FedAvg

        Returns (aggregated_state_dict, normalised_weights).
        """
        n = len(client_weights_list)
        if trusts is None:
            trusts = [1.0] * n

        raw = np.array([s * t for s, t in zip(client_sizes, trusts)],
                       dtype=np.float64)
        if raw.sum() <= 0:                      # degenerate guard
            raw = np.array(client_sizes, dtype=np.float64)
        p = raw / raw.sum()

        aggregated = copy.deepcopy(client_weights_list[0])
        for key in aggregated:
            aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)

        for weights, factor in zip(client_weights_list, p):
            for key in aggregated:
                aggregated[key] += float(factor) * weights[key].float()

        # Restore integer dtypes (e.g. BatchNorm's num_batches_tracked)
        ref = client_weights_list[0]
        for key in aggregated:
            if not torch.is_floating_point(ref[key]):
                aggregated[key] = aggregated[key].to(ref[key].dtype)

        self.weight_history.append(p.tolist())
        return aggregated, p.tolist()

    # Backwards-compatible v1 alias
    def fedavg_aggregate(self, client_weights_list, client_sizes):
        agg, _ = self.aggregate(client_weights_list, client_sizes, trusts=None)
        return agg

    def update_global_model(self, aggregated_weights):
        self.global_model.load_state_dict(aggregated_weights)
        self.global_model.eval()

    # ── Decision-threshold calibration ────────────────────────────────────────
    def _predict_proba(self, X):
        self.global_model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=512, shuffle=False)
        out = []
        with torch.no_grad():
            for (X_b,) in loader:
                out.extend(torch.sigmoid(
                    self.global_model(X_b.to(self.device))).cpu().numpy())
        return np.asarray(out)

    def calibrate_threshold(self, X_val, y_val, objective="f1"):
        """
        Choose the decision threshold on VALIDATION data.

        Why this is necessary, not a convenience
        ────────────────────────────────────────
        DP-SGD injects Gaussian noise into every gradient step, which shrinks the
        magnitude of the learned logits and pulls predicted probabilities toward
        the base rate. At a 0.17% fraud rate the entire predicted distribution can
        sit below 0.5, giving recall = 0.0 even when ROC-AUC exceeds 0.99 — the
        ranking is excellent but the fixed cut-point is in the wrong place.
        Reporting recall at a hard 0.5 would therefore measure the calibration of
        the sigmoid, not the quality of the detector.

        The threshold is selected on validation data and then FROZEN before the
        test set is touched, so no test information leaks into the choice.

        objective:
          "f1"      — maximise validation F1 (default)
          "recall@p"— highest recall subject to precision >= p, the framing an
                      actual fraud team uses (alert budget constrained)
        """
        if X_val is None or len(np.unique(y_val)) < 2:
            self.threshold = 0.5
            return 0.5

        probs = self._predict_proba(X_val)
        candidates = np.unique(np.quantile(probs, np.linspace(0.5, 0.99999, 400)))

        best_thr, best_score = 0.5, -1.0
        for thr in candidates:
            preds = (probs >= thr).astype(int)
            if preds.sum() == 0:
                continue
            if objective.startswith("recall@"):
                p_min = float(objective.split("@")[1])
                prec  = precision_score(y_val, preds, zero_division=0)
                score = recall_score(y_val, preds, zero_division=0) \
                        if prec >= p_min else -1.0
            else:
                score = f1_score(y_val, preds, zero_division=0)
            if score > best_score:
                best_score, best_thr = score, float(thr)

        self.threshold = best_thr
        if self.config.get("verbose", True):
            print(f"[Server] threshold calibrated on validation: "
                  f"{best_thr:.6f} (val {objective}={best_score:.4f})")
        return best_thr

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate_global(self, X_test, y_test, round_num=None, save_best=True,
                        X_val=None, y_val=None):
        """
        Evaluate the global model on the held-out test set.

        PR-AUC is reported alongside ROC-AUC: at a fraud rate near 0.17%,
        ROC-AUC is optimistic and compresses differences between models, so
        average precision is the more informative headline number.
        """
        # Recalibrate the threshold on validation data before each evaluation.
        if X_val is not None:
            self.calibrate_threshold(X_val, y_val,
                                     objective=self.config.get("threshold_objective", "f1"))
        thr = getattr(self, "threshold", 0.5)

        probs  = self._predict_proba(X_test)
        labels = np.asarray(y_test)
        preds  = (probs >= thr).astype(int)

        metrics = {
            "round":     round_num,
            "threshold": float(thr),
            "accuracy":  accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "f1":        f1_score(labels, preds, zero_division=0),
            "auc":       roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5,
            "pr_auc":    average_precision_score(labels, probs)
                         if len(set(labels)) > 1 else 0.0,
            # Kept for transparency: what a naive fixed cut-point would report.
            "f1_at_0.5": f1_score(labels, (probs >= 0.5).astype(int),
                                  zero_division=0),
        }

        # Validation F1 at the same threshold — this, not test F1, is what
        # selects the "best round" below. Picking the round by its TEST score
        # is model selection on the test set and inflates every federated
        # number in the paper; the threshold was already calibrated on
        # validation, so scoring validation here is free.
        val_f1 = None
        if X_val is not None and len(np.unique(y_val)) > 1:
            val_probs = self._predict_proba(X_val)
            val_preds = (val_probs >= thr).astype(int)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
        metrics["val_f1"] = val_f1

        if self.config.get("verbose", True):
            tag = f"Round {round_num}" if round_num is not None else "Final"
            val_str = f" val_f1={val_f1:.4f}" if val_f1 is not None else ""
            print(f"[Server] {tag}: recall={metrics['recall']:.4f} "
                  f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} "
                  f"pr_auc={metrics['pr_auc']:.4f} thr={thr:.4g}{val_str}")

        # Best-model selection uses validation F1 (falls back to test F1 only
        # when no validation set was supplied at all).
        selection_score = val_f1 if val_f1 is not None else metrics["f1"]
        if save_best and selection_score > self.best_f1:
            self.best_f1      = selection_score
            self.best_weights = get_model_weights(self.global_model)
            torch.save(self.best_weights,
                       os.path.join(MODEL_DIR, "best_global_model.pt"))

        self.round_metrics.append(metrics)
        return metrics

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_training_curves(self, filename="federated_training_curves.png"):
        if not self.round_metrics:
            return
        rounds  = [m["round"] for m in self.round_metrics]
        names   = ["accuracy", "precision", "recall", "f1", "auc", "pr_auc"]
        labels  = ["Accuracy", "Precision", "Recall", "F1-Score",
                   "ROC-AUC", "PR-AUC"]
        colors  = ["#3B82F6", "#F59E0B", "#EF4444", "#10B981",
                   "#8B5CF6", "#14B8A6"]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor("#0f1117")
        axes = axes.flatten()

        for ax, name, label, color in zip(axes, names, labels, colors):
            vals = [m.get(name, 0) for m in self.round_metrics]
            ax.plot(rounds, vals, color=color, linewidth=2.5, marker="o", markersize=5)
            ax.fill_between(rounds, vals, alpha=0.15, color=color)
            ax.set_title(label, color="white", fontsize=12)
            ax.set_xlabel("Federated Round", color="white", fontsize=9)
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.set_ylim(0, 1.05)

        plt.suptitle("Federated Learning — Global Model Training Curves",
                     color="white", fontsize=15, y=1.01)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_trust_dynamics(self, filename="adaptive_trust_dynamics.png",
                            drift_round=None):
        """
        THE key figure for contribution #1: per-client trust and aggregation
        weight over rounds, with the drift-injection round marked. A convincing
        version shows the drifting client's weight collapsing right after the
        injection and recovering afterwards.
        """
        if not self.drift_records:
            return

        cids = sorted({r["client_id"] for r in self.drift_records})
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor("#0f1117")
        cmap = plt.get_cmap("tab10")

        panels = [
            ("loss",     "Mean local loss",       axes[0]),
            ("ph_stat",  "Page-Hinkley statistic", axes[1]),
            ("trust",    "Trust score",            axes[2]),
        ]

        for key, title, ax in panels:
            for i, cid in enumerate(cids):
                recs = [r for r in self.drift_records if r["client_id"] == cid]
                xs   = [r["round"] for r in recs]
                ys   = [r[key] for r in recs]
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                        color=cmap(i % 10), label=f"Client {cid}")
                if key == "trust":
                    alarms = [(r["round"], r["trust"]) for r in recs if r["detected"]]
                    if alarms:
                        ax.scatter(*zip(*alarms), marker="x", s=70,
                                   color=cmap(i % 10), zorder=5)

            if key == "ph_stat":
                thr = self.config.get("drift_detector", {}).get("threshold", 0.05)
                ax.axhline(thr, color="#EF4444", linestyle="--", alpha=0.8,
                           label="PH threshold")
            if drift_round:
                ax.axvline(drift_round, color="#F59E0B", linestyle=":",
                           linewidth=2, alpha=0.9, label="drift injected")

            ax.set_title(title, color="white", fontsize=13)
            ax.set_xlabel("Federated Round", color="white")
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=8)

        plt.suptitle("Adaptive FedAvg — Drift Detection and Trust Dynamics",
                     color="white", fontsize=15, y=1.02)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def plot_confusion_matrix(self, X_test, y_test,
                              filename="confusion_matrix.png"):
        self.global_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.global_model(
                torch.tensor(X_test, dtype=torch.float32).to(self.device))
            ).cpu().numpy()
        thr = getattr(self, "threshold", 0.5)
        cm = confusion_matrix(y_test, (probs >= thr).astype(int))

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Fraud", "Fraud"], color="white")
        ax.set_yticklabels(["Non-Fraud", "Fraud"], color="white")
        ax.set_xlabel("Predicted", color="white", fontsize=12)
        ax.set_ylabel("Actual", color="white", fontsize=12)
        ax.set_title("Confusion Matrix — Global Federated Model",
                     color="white", fontsize=13)
        ax.tick_params(colors="white")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def save_final_model(self):
        path = os.path.join(MODEL_DIR, "final_global_model.pt")
        torch.save(get_model_weights(self.global_model), path)
        return path
