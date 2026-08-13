"""
stats_utils.py
==============
Multi-seed aggregation and significance testing  (contribution #5).

Why v1's numbers were not publishable as they stood
───────────────────────────────────────────────────
v1 reported a single run at seed 42. With ~98 fraud records in the test set,
one additional true positive moves recall by roughly one point, so run-to-run
variation from initialisation, SMOTE synthesis and DP noise is comparable to
the differences being claimed between methods. A single run cannot distinguish
"our method is better" from "this seed was lucky."

v2 repeats every headline configuration across N seeds and reports
mean +/- std, plus a paired test against the chosen reference method.

Paired, not unpaired: every method sees the SAME seeds, hence the same data
partition, the same resampling and the same initialisation stream. Pairing on
seed removes that shared variance and is substantially more powerful at the
small N (3) that compute allows.

Honest caveat to carry into the paper: with N = 3 the Wilcoxon signed-rank test
cannot produce a p-value below 0.25, so it is reported for completeness only.
The paired t-test is the usable test at this N, and even it rests on a
normality assumption that 3 points cannot verify. Treat these as indicative,
and say so in the text. If a reviewer pushes back, the fix is more seeds
(N >= 10), not a different test.
"""

import numpy as np

try:
    from scipy import stats as scipy_stats
    SCIPY = True
except ImportError:                                        # pragma: no cover
    SCIPY = False


def aggregate_runs(runs, keys=None):
    """
    runs: list of metric dicts, one per seed.

    Returns {key: {mean, std, sem, ci95_lo, ci95_hi, n, values}}.
    """
    if not runs:
        return {}
    keys = keys or [k for k, v in runs[0].items() if isinstance(v, (int, float))]

    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if k in r and r[k] is not None],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        n    = len(vals)
        mean = float(vals.mean())
        # ddof=1: sample std. With n=1 this is undefined, so report 0.
        std  = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem  = std / np.sqrt(n) if n > 1 else 0.0

        if n > 1 and SCIPY:
            tcrit = scipy_stats.t.ppf(0.975, df=n - 1)
            half  = tcrit * sem
        else:
            half = 0.0

        out[k] = {
            "mean": mean, "std": std, "sem": sem,
            "ci95_lo": mean - half, "ci95_hi": mean + half,
            "n": n, "values": vals.tolist(),
        }
    return out


def paired_test(a, b, alternative="two-sided"):
    """
    Paired comparison of two methods evaluated on the same seeds.

    Returns t-test and Wilcoxon results plus Cohen's d_z effect size.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b

    res = {
        "n": int(n),
        "mean_diff": float(diff.mean()) if n else float("nan"),
        "t_stat": float("nan"), "t_p": float("nan"),
        "w_stat": float("nan"), "w_p": float("nan"),
        "cohens_dz": float("nan"),
    }
    if n < 2 or not SCIPY:
        return res

    sd = diff.std(ddof=1)
    if sd > 0:
        res["cohens_dz"] = float(diff.mean() / sd)
        try:
            t, p = scipy_stats.ttest_rel(a, b, alternative=alternative)
            res["t_stat"], res["t_p"] = float(t), float(p)
        except Exception:
            pass
        try:
            w, p = scipy_stats.wilcoxon(a, b, alternative=alternative)
            res["w_stat"], res["w_p"] = float(w), float(p)
        except Exception:
            # Wilcoxon needs n >= ~6 for a meaningful two-sided p-value
            pass
    else:
        # Zero variance in the differences: identical or perfectly constant gap
        res["cohens_dz"] = float("inf") if diff.mean() != 0 else 0.0

    return res


def fmt(agg_entry, decimals=4):
    """Format an aggregate as 'mean ± std' for tables."""
    if not agg_entry:
        return "—"
    return f"{agg_entry['mean']:.{decimals}f} ± {agg_entry['std']:.{decimals}f}"


def summarise_table(results_by_method, keys=None, reference=None):
    """
    Build a comparison table across methods.

    results_by_method: {method_name: [metric_dict_per_seed, ...]}
    reference:         method name to run paired tests against

    Returns {method: {"agg": {...}, "vs_reference": {metric: test_dict}}}
    """
    keys = keys or ["precision", "recall", "f1", "auc", "pr_auc"]
    table = {}

    for method, runs in results_by_method.items():
        table[method] = {"agg": aggregate_runs(runs, keys), "vs_reference": {}}

    if reference and reference in results_by_method:
        ref_runs = results_by_method[reference]
        for method, runs in results_by_method.items():
            if method == reference:
                continue
            for k in keys:
                a = [r[k] for r in ref_runs if k in r]
                b = [r[k] for r in runs if k in r]
                if len(a) >= 2 and len(b) >= 2:
                    table[method]["vs_reference"][k] = paired_test(a, b)

    return table


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ours  = [{"f1": v} for v in rng.normal(0.80, 0.01, 3)]
    other = [{"f1": v} for v in rng.normal(0.74, 0.01, 3)]
    tbl = summarise_table({"ours": ours, "baseline": other}, keys=["f1"],
                          reference="ours")
    print("ours     :", fmt(tbl["ours"]["agg"]["f1"]))
    print("baseline :", fmt(tbl["baseline"]["agg"]["f1"]))
    print("paired   :", tbl["baseline"]["vs_reference"]["f1"])
