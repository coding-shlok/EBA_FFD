"""
scaling.py
==========
Scalability study across 4, 8, 16 and 32 simulated banks  (contribution #6).

What this experiment actually measures
──────────────────────────────────────
Adding banks to a federation with a FIXED total dataset does two opposing
things:

  (+) more diverse data contributes to each aggregation step
  (−) each bank holds less data, so per-bank models are weaker, per-bank DP
      noise is HIGHER (sigma is calibrated per bank and grows as n_k shrinks —
      see privacy.py), and non-IID skew between banks worsens

This is therefore not a pure throughput benchmark. It is a statement about how
the method degrades under realistic fragmentation, which is the regime an
actual banking consortium sits in.

The hard constraint nobody can design around
────────────────────────────────────────────
The dataset contains 492 fraud records total; roughly 350 survive into the
training split. Across 32 banks that is ~11 fraud records per bank on average,
and under a Dirichlet(0.5) draw the poorest banks would get zero without the
floor enforced in data_loader. Two consequences must be stated plainly in the
paper rather than buried:

  1. At 16 and 32 banks, several clients cannot support KMeansSMOTE and fall
     back to SMOTE or to no resampling. preprocessing.py records this per
     client and the runner reports the breakdown.
  2. Any degradation observed at 32 banks confounds "the method scales poorly"
     with "there is almost no fraud data per bank". Do not claim the former
     without acknowledging the latter. The clean way to separate them is a
     second, larger dataset (IEEE-CIS) — worth flagging as future work.

Reported alongside accuracy metrics:
  - wall-clock time per round
  - mean per-bank sigma (rises as banks shrink)
  - number of clients that hit the balancing fallback
  - number of clients that hit the fraud floor
"""

import time

from config import get_config
from data_loader import load_federated_data
from training_loop import run_federated_training


DEFAULT_SCALES = [4, 8, 16, 32]


def run_scaling_study(base_config, scales=None, seeds=(42,), verbose=True):
    """
    Sweep the number of clients.

    Each scale needs its OWN partition (the partition is a function of
    num_clients), so no data cache is shared across scales — only across seeds
    within a scale, where it must not be shared either since the seed drives the
    partition. Data is therefore reloaded per (scale, seed).

    Returns {num_clients: [metrics_per_seed, ...]}.
    """
    scales = scales or DEFAULT_SCALES
    out = {}

    for k in scales:
        runs = []
        for seed in seeds:
            if verbose:
                print(f"\n[scaling] === {k} clients, seed {seed} ===")

            cfg = get_config(**base_config)
            cfg.update({
                "num_clients": k,
                "seed":        seed,
                "visualize":   False,
                "verbose":     False,
            })

            t0 = time.time()
            try:
                data = load_federated_data(cfg)
                results, server, clients, _, _ = run_federated_training(
                    cfg, X_cache=data)

                m = dict(results["best_metrics"])
                m.update({
                    "num_clients":     k,
                    "seed":            seed,
                    "elapsed_sec":     results["elapsed_sec"],
                    "sec_per_round":   results["elapsed_sec"] / max(1, cfg["num_rounds"]),
                    "mean_sigma":      _mean([r["sigma"] for r in results["privacy_plan"]]),
                    "max_sigma":       max((r["sigma"] for r in results["privacy_plan"]),
                                           default=0.0),
                    "mean_epsilon":    _mean(results["epsilon_spent"]),
                    "min_client_size": min(results["client_sizes"]),
                    "max_client_size": max(results["client_sizes"]),
                    "clients_topped_up": results["partition_meta"].get(
                                            "clients_topped_up", 0),
                    "balancing_fallbacks": sum(
                        1 for b in results["balancing_info"] if b.get("fallback")),
                    "kmeans_smote_used": sum(
                        1 for b in results["balancing_info"]
                        if b.get("applied", "").startswith("kmeans")),
                })
                runs.append(m)

                if verbose:
                    print(f"[scaling] {k:>2} clients: f1={m.get('f1', 0):.4f} "
                          f"recall={m.get('recall', 0):.4f} "
                          f"pr_auc={m.get('pr_auc', 0):.4f} | "
                          f"sigma={m['mean_sigma']:.3f} | "
                          f"{m['sec_per_round']:.1f}s/round | "
                          f"KMeansSMOTE on {m['kmeans_smote_used']}/{k} banks")

            except Exception as e:
                print(f"[scaling] !! {k} clients seed {seed} failed: "
                      f"{type(e).__name__}: {e}")

            if verbose:
                print(f"[scaling] wall clock {time.time() - t0:.1f}s")

        out[k] = runs

    return out


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v and v != float("inf")]
    return float(sum(vals) / len(vals)) if vals else 0.0


if __name__ == "__main__":
    cfg = get_config(subsample_frac=0.05, num_rounds=2, local_epochs=1,
                     verbose=False, visualize=False, min_fraud_per_client=5)
    res = run_scaling_study(cfg, scales=[2, 4], seeds=(42,))
    for k, runs in res.items():
        if runs:
            print(f"{k} clients -> f1={runs[0].get('f1', 0):.4f}")
