"""
main.py
=======
Entry point for a SINGLE EBA-FFD run: data -> balancing -> federated training
with DP + adaptive aggregation -> explainability.

For the full experimental suite that produces every table and figure in the
paper (baselines, ablations, drift study, scaling, multi-seed statistics), use:

    python run_all_experiments.py --preset full

This script is the "show me it works once" path; run_all_experiments.py is the
"produce the paper" path.
"""

import argparse
import os

from config import get_config
from training_loop import run_federated_training, save_results
from explainability import run_explainability


def main():
    ap = argparse.ArgumentParser(description="EBA-FFD v2 — single run")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=3.0,
                    help="per-bank DP budget; use --no-dp to disable privacy")
    ap.add_argument("--no-dp", action="store_true",
                    help="disable differential privacy (NO formal guarantee)")
    ap.add_argument("--aggregation", default="adaptive",
                    choices=["adaptive", "fedavg", "fedprox"])
    ap.add_argument("--drift", action="store_true",
                    help="inject concept drift to demonstrate adaptive aggregation")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="fraction of the dataset to use (e.g. 0.1 for a fast test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-explain", action="store_true")
    args = ap.parse_args()

    config = get_config(
        num_rounds     = args.rounds,
        num_clients    = args.clients,
        local_epochs   = args.epochs,
        target_epsilon = args.epsilon,
        dp_mode        = "none" if args.no_dp else "opacus",
        aggregation    = args.aggregation,
        drift_enabled  = args.drift,
        subsample_frac = args.subsample,
        seed           = args.seed,
        visualize      = True,
        verbose        = True,
    )
    if args.drift:
        config["drift_round"] = max(2, args.rounds // 2)

    results, server, clients, data, resampled = run_federated_training(config)

    save_results(results, os.path.join(
        os.path.dirname(__file__), "outputs", "training_summary.json"))

    # ── Explainability on the best global model ───────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "outputs",
                              "models", "best_global_model.pt")
    if not args.skip_explain and os.path.exists(model_path):
        try:
            run_explainability(model_path, data["X_test"], data["y_test"],
                               data["feature_names"],
                               input_dim=data["X_test"].shape[1],
                               device=config["device"], config=config)
        except Exception as e:
            print(f"[main] explainability skipped: {type(e).__name__}: {e}")

    best = results["best_metrics"]
    print(f"\n{'='*66}")
    print(f"  EBA-FFD v2 complete")
    print(f"  Recall   : {best.get('recall', 0):.4f}")
    print(f"  F1       : {best.get('f1', 0):.4f}")
    print(f"  PR-AUC   : {best.get('pr_auc', 0):.4f}   <- headline metric")
    print(f"  ROC-AUC  : {best.get('auc', 0):.4f}")
    if not args.no_dp:
        eps = results["epsilon_spent"]
        print(f"  Privacy  : eps = {max(eps):.2f} per bank (target {args.epsilon})")
    else:
        print(f"  Privacy  : NONE — no formal guarantee")
    print(f"  Outputs  : outputs/  |  Tables: results/")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
