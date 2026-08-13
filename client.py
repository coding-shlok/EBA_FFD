"""
client.py
=========
Federated client — local training under true DP-SGD.

Each client represents a bank that:
  1. Receives global model weights from the server
  2. Trains on its LOCAL dataset (raw data never leaves the client)
  3. Trains under DP-SGD: per-sample gradient clipping + calibrated Gaussian
     noise, with an RDP accountant tracking the realised (epsilon, delta)
  4. Returns updated weights, its mean local loss (the drift signal), and its
     spent privacy budget

Changes in v2
─────────────
1. REAL differential privacy. v1 clipped minibatch gradients and then added
   noise to the final weights — two disconnected operations from which no
   (epsilon, delta) guarantee follows. v2 delegates to Opacus: per-sample
   clipping, noise on the summed clipped gradients, Poisson subsampling, and a
   composed RDP accounting across every round. See privacy.py.

2. The local model is REBUILT from the received global weights each round
   rather than being wrapped in place. Opacus' GradSampleModule installs
   backward hooks on the module it wraps; re-wrapping the same instance every
   round stacks hooks and corrupts the per-sample gradients. Rebuilding is
   semantically identical under FedAvg (the client is overwritten by the global
   model at the start of every round anyway) and keeps the DP machinery clean.

3. FedProx support. With mu > 0 the local objective gains a proximal term
       L_prox = L + (mu/2) * ||w - w_global||^2
   which restrains client drift on non-IID data (Li et al., 2020). This is the
   federated baseline for contribution #3.

4. The mean local training loss is returned every round. This scalar is the
   only thing the drift detector consumes, so drift detection costs no extra
   privacy budget.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from model import build_model, build_loss, get_model_weights
import privacy as dp


class FederatedClient:
    """
    A simulated federated client (bank).

    Responsibilities:
      - hold a local dataset that never leaves the client
      - receive global weights, train locally under DP-SGD
      - return updated weights + drift signal + privacy accounting
    """

    def __init__(self, client_id, X, y, input_dim=30, config=None, device="cpu"):
        config = config or {}
        self.client_id    = client_id
        self.input_dim    = input_dim
        self.config       = config
        self.device       = device

        self.batch_size   = config.get("batch_size", 256)
        self.local_epochs = config.get("local_epochs", 3)
        self.lr           = config.get("lr", 1e-3)
        self.max_grad_norm= config.get("max_grad_norm", 1.0)
        self.dp_mode      = config.get("dp_mode", "opacus")
        self.fedprox_mu   = config.get("fedprox_mu", 0.0) \
                            if config.get("aggregation") == "fedprox" else 0.0

        self.dataloader = self._build_dataloader(X, y)
        self.n_samples  = len(X)
        self.n_fraud    = int((y == 1).sum())

        self.criterion  = build_loss(config)

        # DP state — sigma is filled in by set_privacy() from the server's plan
        self.sigma       = 0.0
        self.delta       = dp.compute_delta(self.n_samples)
        self.accountant  = None      # persists across rounds so epsilon composes
        self.epsilon_spent = 0.0

        self.global_weights = None
        self.drifted        = False
        self.history = {"loss": [], "f1": [], "recall": [], "epsilon": []}

        if config.get("verbose", True):
            print(f"[Client {client_id}] {self.n_samples:,} samples "
                  f"({self.n_fraud} fraud) | lr={self.lr} | mu={self.fedprox_mu}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    def _build_dataloader(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size,
                          shuffle=True, drop_last=False, num_workers=0)

    def set_privacy(self, sigma, delta):
        """Install this client's individually calibrated noise multiplier."""
        self.sigma = float(sigma)
        self.delta = float(delta)

    def receive_global_weights(self, global_weights):
        """Store the broadcast global weights for this round."""
        self.global_weights = copy.deepcopy(global_weights)

    # ── Local training ────────────────────────────────────────────────────────
    def local_train(self):
        """
        Run local_epochs of DP-SGD on the client's private data.

        Returns (updated_weights, metrics).
        metrics["loss"] is the drift signal consumed by the server.
        """
        model = build_model(self.input_dim, self.device, self.config)
        if self.global_weights is not None:
            model.load_state_dict(self.global_weights)
        model.train()

        # Snapshot of the global model, for the FedProx proximal term
        prox_ref = None
        if self.fedprox_mu > 0 and self.global_weights is not None:
            ref_model = build_model(self.input_dim, self.device, self.config)
            ref_model.load_state_dict(self.global_weights)
            prox_ref = [p.detach().clone() for p in ref_model.parameters()]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                      weight_decay=1e-4)

        loader = self.dataloader
        engine = None
        use_dp = (self.dp_mode == "opacus" and self.sigma > 0)

        if use_dp:
            model = dp.validate_and_fix_model(model)
            model, optimizer, loader, engine = dp.make_private(
                model, optimizer, self.dataloader,
                sigma=self.sigma, max_grad_norm=self.max_grad_norm,
                accountant=self.accountant,
            )
            self.accountant = engine.accountant   # carry the budget forward

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.local_epochs))

        epoch_losses = []
        last_preds, last_labels, last_probs = [], [], []

        for epoch in range(self.local_epochs):
            batch_losses = []
            ep_preds, ep_labels, ep_probs = [], [], []

            for X_batch, y_batch in loader:
                # Poisson subsampling can emit empty batches; skip them.
                if X_batch.numel() == 0 or X_batch.size(0) == 0:
                    continue

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = self.criterion(logits, y_batch)

                # FedProx proximal term
                if prox_ref is not None:
                    prox = sum(((p - r) ** 2).sum()
                               for p, r in zip(model.parameters(), prox_ref))
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                loss.backward()

                # Without Opacus we still clip, but note this is minibatch
                # clipping and carries NO privacy guarantee — it is only here
                # for optimisation stability in the dp_mode="none" ablation.
                if not use_dp:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             max_norm=self.max_grad_norm)

                optimizer.step()
                batch_losses.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu().numpy()
                    ep_preds.extend((probs > 0.5).astype(int))
                    ep_labels.extend(y_batch.cpu().numpy())
                    ep_probs.extend(probs)

            scheduler.step()
            if batch_losses:
                epoch_losses.append(float(np.mean(batch_losses)))
            if epoch == self.local_epochs - 1:
                last_preds, last_labels, last_probs = ep_preds, ep_labels, ep_probs

        # ── Privacy accounting ────────────────────────────────────────────────
        if use_dp and engine is not None:
            self.epsilon_spent = dp.spent_epsilon(engine, self.delta)

        metrics = self._compute_metrics(last_labels, last_preds, last_probs)
        metrics["loss"]    = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        metrics["epsilon"] = self.epsilon_spent
        metrics["delta"]   = self.delta
        metrics["sigma"]   = self.sigma
        metrics["n_samples"] = self.n_samples

        for k in ("loss", "f1", "recall", "epsilon"):
            self.history[k].append(metrics.get(k, 0.0))

        if self.config.get("verbose", True):
            eps_str = f"eps={self.epsilon_spent:.2f}" if use_dp else "eps=inf(no DP)"
            print(f"[Client {self.client_id}] loss={metrics['loss']:.4f} | "
                  f"recall={metrics.get('recall', 0):.4f} | "
                  f"f1={metrics.get('f1', 0):.4f} | {eps_str}")

        # Unwrap Opacus before handing weights to the server
        return get_model_weights(model), metrics

    def _compute_metrics(self, y_true, y_pred, y_prob):
        try:
            if len(y_true) == 0:
                return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0.5}
            return {
                "accuracy":  accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall":    recall_score(y_true, y_pred, zero_division=0),
                "f1":        f1_score(y_true, y_pred, zero_division=0),
                "auc":       roc_auc_score(y_true, y_prob)
                             if len(set(y_true)) > 1 else 0.5,
            }
        except Exception:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0.5}


# ── Client Factory ─────────────────────────────────────────────────────────────
def build_clients(resampled_clients, input_dim=30, device="cpu", config=None):
    """
    Instantiate one FederatedClient per partition, then install each bank's
    individually calibrated DP noise multiplier.
    """
    config  = config or {}
    clients = [
        FederatedClient(cd["client_id"], cd["X"], cd["y"],
                        input_dim=input_dim, config=config, device=device)
        for cd in resampled_clients
    ]

    plan = privacy_plan_for(clients, config)
    for c, rec in zip(clients, plan):
        c.set_privacy(rec["sigma"], rec["delta"])

    return clients, plan


def privacy_plan_for(clients, config):
    """Build the per-bank DP calibration table for the given clients."""
    sizes = [c.n_samples for c in clients]
    plan  = dp.build_privacy_plan(sizes, config)
    if config.get("verbose", True):
        dp.print_privacy_plan(plan, config.get("target_epsilon"))
    return plan


if __name__ == "__main__":
    print("[client] Module OK — import and use via training_loop.py")
