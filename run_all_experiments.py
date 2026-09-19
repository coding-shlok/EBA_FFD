"""
run_all_experiments.py
======================
Unattended experiment runner  (contribution #7).

One command produces every table and figure in the paper:

    python run_all_experiments.py --preset full

Presets
───────
  smoke   ~5 min    2% of data, 2 rounds, 1 seed, 2 scales.
                    Proves the pipeline runs. Numbers are meaningless.
  quick   ~1-3 h    20% of data, 5 rounds, 2 seeds. Sanity-check trends
                    before committing to the full run.
  full    many h    100% of data, 10 rounds, 3 seeds, scales 4/8/16/32.
                    This is the configuration to report.

Everything is checkpointed to results/checkpoints/. Re-running skips stages
already complete, so a crash at hour six does not cost the first five. Use
--force to recompute, or --only to run a single stage.

RUNTIME WARNING
───────────────
DP-SGD with a DPLSTM is roughly an order of magnitude slower than ordinary
training: per-sample gradients are materialised for every parameter, and the
recurrent block cannot use cuDNN's fused kernels. On CPU the full preset is
measured in days, not hours. Use a GPU, and run --only scaling last.
"""

import argparse
import json
import os
import sys
import time
import traceback

from config import get_config
from data_loader import load_federated_data
from preprocessing import preprocess_clients
from training_loop import run_federated_training, _serialisable

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


PRESETS = {
    "smoke": dict(subsample_frac=0.02, num_rounds=2, local_epochs=1,
                  num_clients=3, seeds=[42], scales=[2, 4],
                  min_fraud_per_client=5),
    "quick": dict(subsample_frac=0.20, num_rounds=5, local_epochs=2,
                  num_clients=4, seeds=[42, 43], scales=[4, 8],
                  min_fraud_per_client=10),
    "full":  dict(subsample_frac=1.00, num_rounds=10, local_epochs=3,
                  num_clients=4, seeds=[42, 43, 44], scales=[4, 8, 16, 32],
                  min_fraud_per_client=10),
}

STAGES = ["main", "baselines", "ablations", "drift", "scaling"]


# ── Checkpointing ─────────────────────────────────────────────────────────────
def ckpt_path(stage, preset):
    return os.path.join(CKPT_DIR, f"{preset}_{stage}.json")


def load_ckpt(stage, preset):
    p = ckpt_path(stage, preset)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_ckpt(stage, preset, data):
    with open(ckpt_path(stage, preset), "w") as f:
        json.dump(_serialisable(data), f, indent=2)


def banner(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}", flush=True)


# ── Stages ────────────────────────────────────────────────────────────────────
def stage_main(cfg, preset, seeds):
    """The proposed system across seeds. Also yields the privacy table."""
    banner("STAGE 1/5 — Proposed system (EBA-FFD) across seeds")
    runs, privacy_plan, eps_spent = [], None, None

    for seed in seeds:
        print(f"\n[run_all] EBA-FFD seed {seed}")
        c = get_config(**cfg)
        c.update({"seed": seed, "verbose": False,
                  "visualize": (seed == seeds[0])})
        data = load_federated_data(c)
        results, server, clients, _, _ = run_federated_training(c, X_cache=data)

        m = dict(results["best_metrics"])
        m["seed"] = seed
        runs.append(m)

        if privacy_plan is None:
            privacy_plan = results["privacy_plan"]
            eps_spent    = results["epsilon_spent"]

        print(f"[run_all]   f1={m.get('f1', 0):.4f} "
              f"recall={m.get('recall', 0):.4f} pr_auc={m.get('pr_auc', 0):.4f}")

    return {"runs": runs, "privacy_plan": privacy_plan,
            "epsilon_spent": eps_spent}


def stage_baselines(cfg, preset, seeds):
    """XGBoost, LogReg, local-only, centralized NN, FedAvg, FedProx."""
    from baselines import run_all_baselines
    banner("STAGE 2/5 — Comparison baselines")
    by_method = {}

    for seed in seeds:
        c = get_config(**cfg)
        c.update({"seed": seed, "verbose": False, "visualize": False})
        data = load_federated_data(c)
        resampled, _ = preprocess_clients(data["clients"], c, visualize=False)

        res = run_all_baselines(c, data, resampled, seed=seed, verbose=True)
        for name, metrics in res.items():
            if metrics:
                metrics["seed"] = seed
                by_method.setdefault(name, []).append(metrics)

    return by_method


def stage_ablations(cfg, preset, seeds, with_drift=False):
    from ablations import run_ablations
    label = "with drift" if with_drift else "stationary"
    banner(f"STAGE {'4' if with_drift else '3'}/5 — Ablations ({label})")

    c = get_config(**cfg)
    c.update({"verbose": False, "visualize": False})
    if with_drift:
        c["drift_enabled"] = True
        c["drift_round"]   = max(2, c["num_rounds"] // 2)

    variants = (["full", "no_adaptive"] if with_drift
                else None)   # drift condition only needs the adaptive contrast

    # A fresh partition is drawn PER SEED, matching stage_main's policy — the
    # "full" variant here must be measured under the same data splits as the
    # proposed system in Table 1, or the two tables silently disagree with
    # each other about the same system. Within a seed, every variant still
    # shares one partition, so the ablation varies the component and not the
    # data split (see ablations.py docstring).
    out = {}
    for seed in seeds:
        seed_cfg = dict(c)
        seed_cfg["seed"] = seed
        data = load_federated_data(seed_cfg)
        if with_drift:
            # Target the LARGEST client, not a fixed index. Dirichlet
            # partitioning can hand any fixed index a near-empty client at
            # some seeds (e.g. seed 43 gave "client 0" just 41 of ~40,000
            # training rows) — drifting a client that small is a no-op
            # regardless of severity or aggregation strategy, since its
            # aggregation weight is already negligible either way. The
            # largest client is where adaptive aggregation's protection
            # actually matters, and it stays meaningful at every seed.
            sizes = [len(cd["X"]) for cd in data["clients"]]
            drift_target = max(range(len(sizes)), key=lambda i: sizes[i])
            seed_cfg["drift_clients"] = [drift_target]
        res = run_ablations(seed_cfg, data=data, seeds=(seed,),
                            variants=variants, with_drift=with_drift)
        for variant, runs in res.items():
            out.setdefault(variant, []).extend(runs)
    return out


def stage_scaling(cfg, preset, seeds, scales):
    from scaling import run_scaling_study
    banner("STAGE 5/5 — Scalability across federation size")
    c = get_config(**cfg)
    c.update({"verbose": False, "visualize": False})
    return run_scaling_study(c, scales=scales, seeds=tuple(seeds))


# ── Reporting ─────────────────────────────────────────────────────────────────
def build_reports(main_res, baseline_res, ablation_res, drift_res, scaling_res):
    import reporting as rep
    banner("Building tables and figures")
    sections = []

    if main_res and baseline_res is not None:
        combined = {"eba_ffd": main_res["runs"]}
        combined.update(baseline_res or {})
        labels = {"eba_ffd": "EBA-FFD (ours, DP)",
                  "xgboost": "XGBoost (centralized)",
                  "logreg": "Logistic Regression (centralized)",
                  "local_only": "Local-only (no collaboration)",
                  "centralized_nn": "CNN (centralized)",
                  "fedavg": "FedAvg + DP",
                  "fedprox": "FedProx + DP"}
        rows = rep.report_comparison(combined, labels=labels, reference="eba_ffd")
        rep.plot_comparison_bars(combined, labels=labels)
        sections.append(("Table 1 — Comparison against baselines",
                         rep._md_table(rows, ["Method", "Precision", "Recall",
                                              "F1", "ROC-AUC", "PR-AUC", "p (F1)"])))

    if main_res and main_res.get("privacy_plan"):
        rows = rep.report_privacy(main_res["privacy_plan"],
                                  main_res.get("epsilon_spent"))
        sections.append(("Table 4 — Per-bank differential privacy calibration",
                         rep._md_table(rows, list(rows[0].keys()) if rows else [])))

    if ablation_res:
        rows = rep.report_ablations(ablation_res)
        rep.plot_ablation_deltas(ablation_res)
        sections.append(("Table 2 — Component ablation (stationary)",
                         rep._md_table(rows, ["Variant", "Precision", "Recall",
                                              "F1", "ROC-AUC", "PR-AUC",
                                              r"$\Delta$F1"])))

    if drift_res:
        rows = rep.report_ablations(drift_res, prefix="table2b_ablation_drift")
        sections.append(("Table 2b — Adaptive aggregation under injected drift",
                         rep._md_table(rows, ["Variant", "Precision", "Recall",
                                              "F1", "ROC-AUC", "PR-AUC",
                                              r"$\Delta$F1"])))

    if scaling_res:
        rows = rep.report_scaling(scaling_res)
        rep.plot_scaling_curves(scaling_res)
        sections.append(("Table 3 — Scalability",
                         rep._md_table(rows, list(rows[0].keys()) if rows else [])))

    path = rep.write_summary_markdown(sections)
    print(f"[run_all] summary written -> {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="EBA-FFD v2 experiment runner")
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--only", nargs="*", default=None, choices=STAGES,
                    help="run only these stages")
    ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoints and recompute")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild tables from existing checkpoints")
    args = ap.parse_args()

    preset = args.preset
    p = PRESETS[preset]
    seeds  = p["seeds"]
    scales = p["scales"]

    cfg = {k: v for k, v in p.items() if k not in ("seeds", "scales")}
    cfg["visualize"] = False
    cfg["verbose"]   = False

    stages = args.only if args.only else [s for s in STAGES if s not in args.skip]

    print(f"[run_all] preset={preset} seeds={seeds} scales={scales}")
    print(f"[run_all] stages={stages}")
    if preset == "smoke":
        print("[run_all] NOTE: smoke preset validates the pipeline only. "
              "Its numbers are not meaningful.")

    store = {}
    t0 = time.time()

    if args.report_only:
        for s in STAGES:
            store[s] = load_ckpt(s, preset)
    else:
        runners = {
            "main":      lambda: stage_main(cfg, preset, seeds),
            "baselines": lambda: stage_baselines(cfg, preset, seeds),
            "ablations": lambda: stage_ablations(cfg, preset, seeds, False),
            "drift":     lambda: stage_ablations(cfg, preset, seeds, True),
            "scaling":   lambda: stage_scaling(cfg, preset, seeds, scales),
        }
        for stage in STAGES:
            if stage not in stages:
                store[stage] = load_ckpt(stage, preset)
                continue

            cached = None if args.force else load_ckpt(stage, preset)
            if cached is not None:
                print(f"[run_all] {stage}: using checkpoint "
                      f"(--force to recompute)")
                store[stage] = cached
                continue

            try:
                store[stage] = runners[stage]()
                save_ckpt(stage, preset, store[stage])
            except Exception:
                print(f"[run_all] !! stage '{stage}' failed:\n"
                      f"{traceback.format_exc()}")
                store[stage] = None

    # Scaling checkpoints round-trip integer keys as strings
    sc = store.get("scaling")
    if isinstance(sc, dict):
        sc = {int(k): v for k, v in sc.items()}

    build_reports(store.get("main"), store.get("baselines"),
                  store.get("ablations"), store.get("drift"), sc)

    print(f"\n[run_all] ALL DONE in {(time.time() - t0)/60:.1f} min")
    print(f"[run_all] tables  -> results/")
    print(f"[run_all] figures -> outputs/plots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
