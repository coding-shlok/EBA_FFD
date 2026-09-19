"""
ablations.py
============
Ablation study  (contribution #4).

Each variant differs from the full system in EXACTLY ONE dimension. That is the
whole point: if two things change at once the resulting delta cannot be
attributed. config.get_config() merges overrides onto a single shared base
precisely so this invariant is mechanically enforced rather than maintained by
hand.

Variants
────────
  full             The proposed system: plain MLP head, no CNN, no BiLSTM.
                   The reference row. This was NOT the original default —
                   capacity_sweep.py's first pass only compared CNN-only
                   against CNN+BiLSTM and picked CNN-only, but the "mlp_only"
                   row in this same ablation table kept showing MLP beating
                   CNN-only too. Re-checked at n=3 seeds specifically to rule
                   out a small-sample fluke: MLP beat CNN-only at every seed
                   (42: 0.7416 vs 0.7159, 43: 0.8042 vs 0.7812, 44: 0.7978 vs
                   0.7771 F1) — a consistent ~2-2.5pt gain, not noise. Same
                   underlying cause as cutting the LSTM: fewer parameters is
                   less surface for DP-SGD's per-sample noise to corrupt, and
                   the raw 30-feature vector has no spatial locality for a 1D
                   convolution to exploit either.
  with_cnn         Adds the CNN block back (no BiLSTM) — this is the
                   architecture "full" used to be, before the re-check above.
                   Kept as an ablation so the paper documents why it was cut,
                   rather than silently disappearing, mirroring "with_lstm".
  with_lstm        Adds the CNN and BiLSTM blocks back on top of the MLP —
                   this is the architecture v2 originally shipped with.
                   capacity_sweep.py found it scores 0.15 F1 / 0.09 PR-AUC
                   worse than CNN-only at matched epsilon: DP-SGD noise on the
                   DPLSTM's per-timestep per-sample gradients dominates its
                   own signal, and the dataset has no genuine per-account
                   sequence for it to model anyway (see README "Known
                   limitations" #5).
  lstm_only        Raw features fed to a BiLSTM as a length-D sequence (CNN
                   off, LSTM explicitly on). Isolates the recurrent block
                   without the CNN's feature extraction ahead of it.
  no_balancing     No SMOTE/ENN. Isolates the resampling stage. Expect a large
                   recall drop at a 0.17% base rate.
  with_focal       Adds Focal Loss back in place of plain BCE -- this is the
                   loss function "full" used to be. Full-scale evidence (n=3
                   seeds) showed plain BCE scoring F1 0.8170 vs Focal Loss's
                   0.7936 (+0.0234, plus recall 0.7891 vs 0.7483) -- the
                   largest single gain in this table -- so BCE is now the
                   config.py default and this variant reproduces the
                   discarded one for the record, mirroring "with_cnn".
  no_dp            DP disabled entirely. NOT a competing system — it has no
                   privacy guarantee. Its role is to measure the utility cost of
                   privacy, i.e. the gap between full and no_dp IS the price of
                   epsilon.
  no_adaptive      Vanilla FedAvg instead of drift-aware aggregation. Isolates
                   contribution #1.

NOTE ON ARCHITECTURE VARIANTS: every variant that touches use_cnn/use_lstm
states BOTH flags explicitly rather than overriding just one and relying on
config.py's default for the other. This is deliberate — it keeps each
variant's meaning fixed even if the BASE_CONFIG default architecture changes
later (as it already has once; see config.py), preserving the "exactly one
dimension differs from full" invariant this module is built around.

Reading the table
─────────────────
A component "pulls its weight" when removing it degrades the metric that
component was introduced to improve — not merely when it changes some metric.
Judge no_balancing on recall and PR-AUC, not accuracy, which is ~99.8% for a
model that predicts "never fraud".

no_adaptive deserves care. Under stationary conditions Adaptive FedAvg reduces
to FedAvg by construction, so full ≈ no_adaptive is the EXPECTED result and is
not a failure. The ablation only becomes informative when run with
drift_enabled=True, which is why run_ablations() takes a with_drift flag and
the runner executes both conditions.
"""

import copy

from config import get_config
from training_loop import run_federated_training


ABLATION_VARIANTS = {
    "full": {
        "label": "Full system (proposed: MLP, no CNN/BiLSTM)",
        "overrides": {},
    },
    "with_cnn": {
        "label": "+ CNN block (discarded convolutional variant)",
        "overrides": {"model": {"use_cnn": True, "use_lstm": False}},
    },
    "with_lstm": {
        "label": "+ CNN and BiLSTM blocks (discarded recurrent variant)",
        "overrides": {"model": {"use_cnn": True, "use_lstm": True}},
    },
    "lstm_only": {
        "label": "+ BiLSTM only (no CNN)",
        "overrides": {"model": {"use_cnn": False, "use_lstm": True}},
    },
    "no_balancing": {
        "label": "− KMeans-SMOTE/ENN balancing",
        "overrides": {"balancing": "none"},
    },
    "with_focal": {
        "label": "+ Focal Loss (discarded loss function)",
        "overrides": {"loss": "focal"},
    },
    "no_dp": {
        "label": "− Differential privacy (no guarantee)",
        "overrides": {"dp_mode": "none"},
    },
    "no_adaptive": {
        "label": "− Adaptive aggregation (vanilla FedAvg)",
        "overrides": {"aggregation": "fedavg"},
    },
}


def run_single_ablation(variant, base_config, data=None, seed=42,
                        with_drift=False, verbose=False):
    """
    Run one ablation variant at one seed.

    NOTE ON DATA CACHING: variants that change `balancing` must re-run
    preprocessing, but they can still share the same raw partition. Passing
    `data` (the loader's output dict) reuses the partition while allowing
    preprocessing to differ — which is what we want, since the partition should
    be identical across variants and only the ablated stage should change.
    """
    spec = ABLATION_VARIANTS[variant]

    cfg = get_config(**copy.deepcopy(spec["overrides"]))
    cfg.update({k: v for k, v in base_config.items()
                if k not in spec["overrides"] and k not in ("model",)})
    # Re-apply nested model overrides after the flat update
    if "model" in spec["overrides"]:
        cfg["model"].update(spec["overrides"]["model"])
    if "dp_mode" in spec["overrides"]:
        cfg["dp_mode"] = spec["overrides"]["dp_mode"]
    if "aggregation" in spec["overrides"]:
        cfg["aggregation"] = spec["overrides"]["aggregation"]
    if "balancing" in spec["overrides"]:
        cfg["balancing"] = spec["overrides"]["balancing"]
    if "loss" in spec["overrides"]:
        cfg["loss"] = spec["overrides"]["loss"]

    cfg["seed"]      = seed
    cfg["visualize"] = False
    cfg["verbose"]   = verbose

    # A model without DP may use the faster standard layers
    if cfg["dp_mode"] != "opacus":
        cfg["model"]["dp_lstm"] = False

    if with_drift:
        cfg["drift_enabled"] = True

    results, *_ = run_federated_training(cfg, X_cache=data)
    # Under drift, "best round by validation" structurally always prefers the
    # PRE-drift round — injected corruption can only hurt, so the pre-drift
    # round is the best any variant can ever score, for every aggregation
    # strategy alike, making full vs no_adaptive identical by construction
    # regardless of severity (verified empirically: both selected round 1
    # at every drift level tested). The scientific question here is sustained
    # performance DURING the drift regime, so report the FINAL round instead.
    # Stationary ablations keep best-by-validation, where that bias doesn't apply.
    metrics = dict(results["final_metrics"] if with_drift else results["best_metrics"])
    metrics["variant"]      = variant
    metrics["label"]        = spec["label"]
    metrics["seed"]         = seed
    metrics["elapsed_sec"]  = results["elapsed_sec"]
    metrics["epsilon_mean"] = (
        sum(results["epsilon_spent"]) / len(results["epsilon_spent"])
        if results["epsilon_spent"] else float("inf")
    )
    return metrics


def run_ablations(base_config, data=None, seeds=(42,), variants=None,
                  with_drift=False, verbose=True):
    """
    Run every variant across every seed.

    Returns {variant_name: [metrics_per_seed, ...]}, ready for
    stats_utils.summarise_table(..., reference="full").
    """
    variants = variants or list(ABLATION_VARIANTS.keys())
    out = {v: [] for v in variants}

    for variant in variants:
        for seed in seeds:
            if verbose:
                print(f"\n[ablations] {variant} (seed {seed}"
                      f"{', drift' if with_drift else ''})")
            try:
                m = run_single_ablation(variant, base_config, data=data,
                                        seed=seed, with_drift=with_drift)
                out[variant].append(m)
                if verbose:
                    print(f"[ablations]   f1={m.get('f1', 0):.4f} "
                          f"recall={m.get('recall', 0):.4f} "
                          f"pr_auc={m.get('pr_auc', 0):.4f}")
            except Exception as e:
                print(f"[ablations] !! {variant} seed {seed} failed: "
                      f"{type(e).__name__}: {e}")

    return out


if __name__ == "__main__":
    from data_loader import load_federated_data

    cfg  = get_config(num_clients=3, subsample_frac=0.05, num_rounds=2,
                      local_epochs=1, verbose=False, visualize=False)
    data = load_federated_data(cfg)

    res = run_ablations(cfg, data=data, seeds=(42,),
                        variants=["full", "no_balancing", "with_cnn"])
    for v, runs in res.items():
        if runs:
            print(f"{v:16s} f1={runs[0].get('f1', 0):.4f}")
