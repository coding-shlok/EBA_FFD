"""
tree_baselines_subprocess.py
=============================
Entry point for running xgboost/local_only in a subprocess that never imports
torch (see tree_baselines.py for why this has to be a separate process).

Not meant to be run manually — baselines.py invokes this via subprocess.

    python tree_baselines_subprocess.py <input.npz> <output.json> <seed>

input.npz: X_val, y_val, X_test, y_test, n_clients, and per-client arrays
           named client_{i}_X / client_{i}_y.
output.json: {"xgboost": {...metrics...}, "local_only": {...metrics...}}
"""
import json
import sys

import numpy as np

from tree_baselines import run_xgboost, run_local_only


def main():
    in_path, out_path, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data = np.load(in_path)

    X_val, y_val   = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    n_clients = int(data["n_clients"])
    resampled = [
        {"X": data[f"client_{i}_X"], "y": data[f"client_{i}_y"], "client_id": i}
        for i in range(n_clients)
    ]

    out = {}
    try:
        out["xgboost"] = run_xgboost(resampled, X_val, y_val, X_test, y_test, seed)
    except Exception as e:
        out["xgboost"] = {}
        print(f"[tree_baselines_subprocess] xgboost failed: {e}", file=sys.stderr)

    try:
        out["local_only"] = run_local_only(resampled, X_val, y_val, X_test, y_test, seed)
    except Exception as e:
        out["local_only"] = {}
        print(f"[tree_baselines_subprocess] local_only failed: {e}", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
