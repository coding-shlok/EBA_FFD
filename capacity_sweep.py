"""
capacity_sweep.py
==================
Step 1 experiment: architecture-capacity sweep under DP-SGD, at fixed
epsilon=3.0 (matching the rest of the paper's DP budget).

Table 2's ablation already showed the qualitative signal: a 12k-parameter MLP
beats the 249k-parameter CNN+BiLSTM by 0.15 F1 at matched privacy budget. This
sweep finds where along the capacity axis PR-AUC actually peaks, instead of
jumping straight from "everything" to "nothing" — the two points the existing
ablation table happens to contain.

Two phases:
  Phase A (broad, cheap)  — every candidate architecture, 1 seed, small data.
                            Ranks candidates fast so Phase B doesn't spend
                            DP-SGD's slow wall-clock time on obviously-bad ones.
  Phase B (confirm)       — top candidates + the two known anchors (full,
                            mlp_only), 2 seeds, at the actual `quick` preset
                            settings — directly comparable to results/table1,
                            table2 and results/RESULTS.md.

Usage:
    python capacity_sweep.py --phase a
    python capacity_sweep.py --phase b
    python capacity_sweep.py --phase both        (default)
"""

import argparse
import json
import os
import time

import numpy as np

from config import get_config
from data_loader import load_federated_data
from training_loop import run_federated_training, _serialisable

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# Every candidate keeps dp_mode="opacus", target_epsilon=3.0 (config default) —
# capacity is the only thing varied, exactly as an ablation should be.
VARIANTS = {
    "full_64_64x2":   {},   # baseline: cnn_channels=64, lstm_hidden=64, lstm_layers=2 (config defaults)
    "cnn128_no_lstm": {"use_lstm": False, "cnn_channels": 128},
    "cnn64_no_lstm":  {"use_lstm": False, "cnn_channels": 64},
    "cnn32_no_lstm":  {"use_lstm": False, "cnn_channels": 32},
    "cnn16_no_lstm":  {"use_lstm": False, "cnn_channels": 16},
    "cnn32_lstm16x1": {"cnn_channels": 32, "lstm_hidden": 16, "lstm_layers": 1},
    "cnn64_lstm16x1": {"cnn_channels": 64, "lstm_hidden": 16, "lstm_layers": 1},
    "mlp_only":       {"use_cnn": False, "use_lstm": False},
}

PHASE_A_CFG = dict(subsample_frac=0.10, num_rounds=3, local_epochs=1,
                    num_clients=3, min_fraud_per_client=5)
PHASE_B_CFG = dict(subsample_frac=0.20, num_rounds=5, local_epochs=2,
                    num_clients=4, min_fraud_per_client=10)


def run_variant(name, overrides, base_cfg, seed, data=None):
    cfg = get_config(**base_cfg)
    cfg["model"].update(overrides)
    cfg["seed"]      = seed
    cfg["verbose"]   = False
    cfg["visualize"] = False

    t0 = time.time()
    results, *_ = run_federated_training(cfg, X_cache=data)
    elapsed = time.time() - t0

    m = dict(results["best_metrics"])
    m["variant"]     = name
    m["seed"]        = seed
    m["n_params"]    = results["n_params"]
    m["elapsed_sec"] = elapsed
    return m


def run_phase(cfg_base, variants, seeds, label):
    print(f"\n{'='*70}\n  CAPACITY SWEEP — {label}\n{'='*70}", flush=True)
    out = {}
    for seed in seeds:
        c = dict(cfg_base)
        c["seed"] = seed
        data = load_federated_data(get_config(**c))
        for name, overrides in variants.items():
            print(f"\n[capacity_sweep] {label} | {name} | seed {seed}", flush=True)
            m = run_variant(name, overrides, cfg_base, seed, data=data)
            out.setdefault(name, []).append(m)
            val_f1 = m.get("val_f1") or 0.0
            print(f"[capacity_sweep]   params={m['n_params']:>8,}  "
                  f"f1={m.get('f1', 0):.4f}  val_f1={val_f1:.4f}  "
                  f"pr_auc={m.get('pr_auc', 0):.4f}  ({m['elapsed_sec']:.0f}s)",
                  flush=True)
    return out


def summarise(out):
    rows = []
    for name, runs in out.items():
        if not runs:
            continue
        rows.append({
            "variant":     name,
            "n_params":    runs[0]["n_params"],
            "f1_mean":     float(np.mean([r.get("f1", 0) for r in runs])),
            "pr_auc_mean": float(np.mean([r.get("pr_auc", 0) for r in runs])),
            "recall_mean": float(np.mean([r.get("recall", 0) for r in runs])),
            "n_seeds":     len(runs),
        })
    rows.sort(key=lambda r: r["pr_auc_mean"], reverse=True)
    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print(f"{'variant':<18}{'params':>10}{'F1':>8}{'PR-AUC':>8}{'recall':>8}{'n':>4}")
    for r in rows:
        print(f"{r['variant']:<18}{r['n_params']:>10,}{r['f1_mean']:>8.4f}"
              f"{r['pr_auc_mean']:>8.4f}{r['recall_mean']:>8.4f}{r['n_seeds']:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=4,
                    help="phase b: how many top phase-a candidates to confirm")
    args = ap.parse_args()

    phase_a_out = None
    if args.phase in ("a", "both"):
        phase_a_out = run_phase(PHASE_A_CFG, VARIANTS, seeds=[42],
                                label="Phase A (broad scan)")
        with open(os.path.join(CKPT_DIR, "capacity_sweep_phase_a.json"), "w") as f:
            json.dump(_serialisable(phase_a_out), f, indent=2)
        rows_a = summarise(phase_a_out)
        print_table(rows_a, "Phase A ranking (by PR-AUC)")

    if args.phase in ("b", "both"):
        if phase_a_out is not None:
            rows_a = summarise(phase_a_out)
            top = [r["variant"] for r in rows_a[:args.top_k]]
            for anchor in ("full_64_64x2", "mlp_only"):
                if anchor not in top:
                    top.append(anchor)
        else:
            top = list(VARIANTS.keys())

        variants_b = {k: VARIANTS[k] for k in top}
        phase_b_out = run_phase(PHASE_B_CFG, variants_b, seeds=[42, 43],
                                label="Phase B (confirm, quick-preset scale)")
        with open(os.path.join(CKPT_DIR, "capacity_sweep_phase_b.json"), "w") as f:
            json.dump(_serialisable(phase_b_out), f, indent=2)
        rows_b = summarise(phase_b_out)
        print_table(rows_b, "Phase B ranking (by PR-AUC) — REPORT THESE NUMBERS")


if __name__ == "__main__":
    main()
