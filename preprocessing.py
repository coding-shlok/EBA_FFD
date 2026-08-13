"""
preprocessing.py
================
Per-client class balancing.

BUG FIXED IN v2 — read this before writing the paper's methods section
─────────────────────────────────────────────────────────────────────
v1 defined `apply_kmeans_smoteenn` TWICE in this file. The first definition was
truncated mid-body; Python silently discarded it and kept the second, which
called plain `SMOTE`. So although the module, the README and the system diagram
all claimed KMeans-SMOTE, every reported v1 number was produced by vanilla
SMOTE + ENN. Any paper written from the v1 code would have misdescribed its own
method — a reviewer reproducing the code would have caught it immediately.

v2 removes the dead definition and actually attempts `KMeansSMOTE`, with an
explicit, LOGGED fallback chain:

    KMeansSMOTE  ->  SMOTE  ->  no resampling

The fallback matters. KMeansSMOTE needs enough minority samples to form
clusters whose minority density clears its threshold; on clients holding only a
handful of fraud records it legitimately cannot run. Rather than failing
silently, each client records which method it actually used, and
`summarise_balancing` reports the breakdown so the paper can state exactly how
often the intended method applied. Under a 32-bank split this is frequently NOT
KMeansSMOTE, and that must be disclosed rather than glossed over.

Strategy:
  1. KMeans clustering locates dense minority (fraud) regions
  2. SMOTE synthesises fraud samples *within* those dense clusters, avoiding
     interpolation across sparse/noisy regions
  3. ENN (Edited Nearest Neighbours) removes borderline majority samples

Ablation switch (config["balancing"]):
  "kmeans_smote_enn" (default) | "smote_enn" | "smote" | "none"
"""

import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from imblearn.over_sampling import KMeansSMOTE, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

PLOT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def apply_balancing(X, y, method="kmeans_smote_enn", random_state=42,
                    k_neighbors=5, sampling_strategy=0.3, verbose=True):
    """
    Balance one client's local dataset.

    Returns (X_res, y_res, info) where info records the method actually applied,
    so downstream reporting can be honest about fallbacks.
    """
    info = {
        "requested":  method,
        "applied":    None,
        "fallback":   None,
        "n_before":   int(len(y)),
        "fraud_before": int((y == 1).sum()),
    }

    if verbose:
        print(f"[preprocessing] Before resampling: {Counter(y)}")

    if method == "none":
        info["applied"] = "none"
        info["n_after"] = int(len(y))
        info["fraud_after"] = int((y == 1).sum())
        if verbose:
            print("[preprocessing] Balancing disabled (ablation).")
        return X.astype(np.float32), y.astype(np.int64), info

    minority = int((y == 1).sum())
    if minority < 2:
        info["applied"]  = "none"
        info["fallback"] = f"only {minority} minority sample(s)"
        info["n_after"]  = int(len(y))
        info["fraud_after"] = minority
        if verbose:
            print(f"[preprocessing] Too few minority samples ({minority}); "
                  f"skipping resampling.")
        return X.astype(np.float32), y.astype(np.int64), info

    k = min(k_neighbors, max(1, minority - 1))
    X_over, y_over = None, None

    # ── Step 1: oversample ────────────────────────────────────────────────────
    if method == "kmeans_smote_enn":
        try:
            kms = KMeansSMOTE(
                sampling_strategy = sampling_strategy,
                random_state      = random_state,
                k_neighbors       = k,
                cluster_balance_threshold = "auto",
            )
            X_over, y_over = kms.fit_resample(X, y)
            info["applied"] = "kmeans_smote"
            if verbose:
                print(f"[preprocessing] After KMeansSMOTE: {Counter(y_over)}")
        except Exception as e:
            info["fallback"] = f"KMeansSMOTE failed: {type(e).__name__}"
            if verbose:
                print(f"[preprocessing] KMeansSMOTE unavailable ({e}); "
                      f"falling back to SMOTE.")

    if X_over is None:
        try:
            sm = SMOTE(sampling_strategy=sampling_strategy,
                       random_state=random_state, k_neighbors=k)
            X_over, y_over = sm.fit_resample(X, y)
            info["applied"] = "smote"
            if verbose:
                print(f"[preprocessing] After SMOTE: {Counter(y_over)}")
        except Exception as e:
            info["applied"]  = "none"
            info["fallback"] = f"SMOTE failed: {type(e).__name__}"
            info["n_after"]  = int(len(y))
            info["fraud_after"] = minority
            if verbose:
                print(f"[preprocessing] SMOTE failed ({e}); skipping resampling.")
            return X.astype(np.float32), y.astype(np.int64), info

    # ── Step 2: ENN cleaning ──────────────────────────────────────────────────
    if method in ("kmeans_smote_enn", "smote_enn"):
        try:
            enn = EditedNearestNeighbours(n_neighbors=3)
            X_res, y_res = enn.fit_resample(X_over, y_over)
            info["applied"] = (info["applied"] or "smote") + "_enn"
            if verbose:
                print(f"[preprocessing] After ENN cleaning: {Counter(y_res)}")
        except Exception as e:
            X_res, y_res = X_over, y_over
            info["fallback"] = (info.get("fallback") or "") + \
                               f" | ENN failed: {type(e).__name__}"
    else:
        X_res, y_res = X_over, y_over

    info["n_after"]     = int(len(y_res))
    info["fraud_after"] = int((y_res == 1).sum())

    if verbose:
        print(f"[preprocessing] Final fraud ratio: {y_res.mean()*100:.2f}%")

    return X_res.astype(np.float32), y_res.astype(np.int64), info


# Backwards-compatible alias for v1 call sites
def apply_kmeans_smoteenn(X, y, random_state=42, k_neighbors=5):
    X_res, y_res, _ = apply_balancing(
        X, y, method="kmeans_smote_enn",
        random_state=random_state, k_neighbors=k_neighbors)
    return X_res, y_res


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_class_distribution(y_before, y_after, client_id=None, save=True):
    """Side-by-side bar charts: class distribution before vs after resampling."""
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(1, 2, figure=fig)

    label         = f"Client {client_id}" if client_id is not None else "Global"
    before_counts = Counter(y_before)
    after_counts  = Counter(y_after)
    categories    = ["Non-Fraud (0)", "Fraud (1)"]
    before_vals   = [before_counts[0], before_counts[1]]
    after_vals    = [after_counts[0],  after_counts[1]]

    for ax, vals, title, colors, ratio_color in [
        (fig.add_subplot(gs[0]), before_vals, f"BEFORE — {label}",
         ["#3B82F6", "#EF4444"], "#EF4444"),
        (fig.add_subplot(gs[1]), after_vals,  f"AFTER  — {label}",
         ["#3B82F6", "#10B981"], "#10B981"),
    ]:
        bars = ax.bar(categories, vals, color=colors, width=0.5,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(title, color="white", fontsize=13, pad=12)
        ax.set_ylabel("Sample Count", color="white")
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"{v:,}", ha="center", color="white", fontsize=10)
        total = max(1, sum(vals))
        ax.text(0.5, 0.95, f"Fraud: {vals[1]/total*100:.3f}%",
                transform=ax.transAxes, ha="center",
                color=ratio_color, fontsize=11)

    plt.suptitle("Class Imbalance: Before vs After Resampling",
                 color="white", fontsize=15, y=1.02)
    plt.tight_layout()

    if save:
        suffix = f"_client{client_id}" if client_id is not None else "_global"
        path = os.path.join(PLOT_DIR, f"class_distribution{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_all_clients_distribution(client_data_raw, client_data_resampled):
    """Multi-client fraud ratio comparison before/after resampling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    n             = len(client_data_raw)
    before_ratios = [d["y"].mean() * 100 for d in client_data_raw]
    after_ratios  = [d["y"].mean() * 100 for d in client_data_resampled]
    client_labels = [f"C{i}" for i in range(n)]

    for ax, ratios, title, color in zip(
        axes, [before_ratios, after_ratios],
        ["Before Resampling", "After Resampling"], ["#EF4444", "#10B981"]
    ):
        bars = ax.bar(client_labels, ratios, color=color,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(title, color="white", fontsize=13)
        ax.set_ylabel("Fraud Ratio (%)", color="white")
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        if n <= 12:
            for bar, val in zip(bars, ratios):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", color="white", fontsize=9)

    plt.suptitle("Per-Client Fraud Ratios — Non-IID Heterogeneity",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "per_client_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Driver ─────────────────────────────────────────────────────────────────────
def preprocess_clients(client_data, config=None, visualize=True):
    """
    Apply balancing to each client's local dataset independently.
    Federated privacy is preserved — clients never share raw data.

    Returns (resampled_clients, balancing_info_list).
    """
    config  = config or {}
    method  = config.get("balancing", "kmeans_smote_enn")
    strat   = config.get("sampling_strategy", 0.3)
    verbose = config.get("verbose", True)

    resampled, infos = [], []
    for client in client_data:
        cid  = client["client_id"]
        X, y = client["X"], client["y"]
        if verbose:
            print(f"\n[preprocessing] === Client {cid} ===")

        y_before = y.copy()
        X_res, y_res, info = apply_balancing(
            X, y, method=method, random_state=42 + cid,
            sampling_strategy=strat, verbose=verbose)
        info["client_id"] = cid
        infos.append(info)

        if visualize and len(client_data) <= 8:
            plot_class_distribution(y_before, y_res, client_id=cid)

        resampled.append({"X": X_res, "y": y_res, "client_id": cid})

    if visualize:
        plot_all_clients_distribution(client_data, resampled)

    if verbose:
        summarise_balancing(infos)

    return resampled, infos


def summarise_balancing(infos):
    """
    Report which balancing method each client ACTUALLY used. This is the number
    the paper should quote — not the configured intent.
    """
    counts = Counter(i["applied"] for i in infos)
    print(f"\n[preprocessing] Balancing methods actually applied across "
          f"{len(infos)} clients:")
    for method, c in counts.most_common():
        print(f"[preprocessing]   {method:22s} {c:>3} client(s)")
    fallbacks = [i for i in infos if i.get("fallback")]
    if fallbacks:
        print(f"[preprocessing]   ({len(fallbacks)} client(s) hit a fallback path)")
    return counts


if __name__ == "__main__":
    from config import get_config
    from data_loader import load_federated_data

    cfg = get_config(num_clients=4, subsample_frac=0.1, min_fraud_per_client=10)
    client_data, X_test, y_test, names, scaler, meta = load_federated_data(cfg)
    resampled, infos = preprocess_clients(client_data, cfg, visualize=False)
    print("\n[preprocessing] Done.")
