"""
reporting.py
============
Turns raw experiment output into the tables and figures a paper actually needs.

Everything is emitted in three formats:
  .csv  — for your own re-analysis
  .tex  — booktabs tables, paste straight into the manuscript
  .md   — for the README / quick reading

Formatting conventions applied throughout:
  - every metric is "mean ± std" across seeds, never a single run
  - the best value per column is bolded in the LaTeX output
  - methods with NO privacy guarantee are marked with a dagger, so a reader
    can never mistake an upper bound for a competitor
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats_utils import aggregate_runs, paired_test, fmt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR    = os.path.join(os.path.dirname(__file__), "outputs", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

METRICS = ["precision", "recall", "f1", "auc", "pr_auc"]
HEADERS = {"precision": "Precision", "recall": "Recall", "f1": "F1",
           "auc": "ROC-AUC", "pr_auc": "PR-AUC"}

# Methods that pool raw data or ignore privacy entirely. NOTE: architecture
# ablations (with_cnn, with_lstm, lstm_only) still run under DP and are NOT
# listed here — only no_dp drops the guarantee.
NO_PRIVACY = {"xgboost", "logreg", "centralized_nn", "local_only", "no_dp"}


# ── Generic writers ───────────────────────────────────────────────────────────
def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _write_text(path, text):
    with open(path, "w") as f:
        f.write(text)
    return path


def _latex_table(rows, columns, caption, label, bold_best=True,
                 higher_is_better=True):
    """Render a booktabs LaTeX table from a list of {col: str_or_float}."""
    best = {}
    if bold_best:
        for c in columns[1:]:
            vals = []
            for r in rows:
                v = r.get(f"_{c}_mean")
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(v)
            if vals:
                best[c] = max(vals) if higher_is_better else min(vals)

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * (len(columns) - 1) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [str(r.get(columns[0], ""))]
        for c in columns[1:]:
            cell = str(r.get(c, "—")).replace("±", r"$\pm$").replace("—", "---")
            mv   = r.get(f"_{c}_mean")
            if c in best and isinstance(mv, (int, float)) and \
               np.isfinite(mv) and abs(mv - best[c]) < 1e-12:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _md_table(rows, columns):
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "—")) for c in columns) + " |")
    return "\n".join(out)


# ── Table 1: main comparison vs baselines ─────────────────────────────────────
def report_comparison(results_by_method, labels=None, reference="eba_ffd",
                      prefix="table1_comparison"):
    """
    results_by_method: {method: [metrics_dict_per_seed, ...]}

    Emits the headline comparison table with paired significance tests against
    the proposed method.
    """
    labels = labels or {}
    rows, csv_rows = [], []

    ref_runs = results_by_method.get(reference, [])

    for method, runs in results_by_method.items():
        if not runs:
            continue
        agg = aggregate_runs(runs, METRICS)
        dagger = r"$^{\dagger}$" if method in NO_PRIVACY else ""
        row = {"Method": labels.get(method, method) + dagger}
        csv_row = {"method": method, "n_seeds": len(runs)}

        for m in METRICS:
            if m in agg:
                row[HEADERS[m]]        = fmt(agg[m])
                row[f"_{HEADERS[m]}_mean"] = agg[m]["mean"]
                csv_row[f"{m}_mean"]   = round(agg[m]["mean"], 6)
                csv_row[f"{m}_std"]    = round(agg[m]["std"], 6)
            else:
                row[HEADERS[m]] = "—"

        # Paired test vs the proposed method, on F1
        if method != reference and ref_runs:
            a = [r["f1"] for r in ref_runs if "f1" in r]
            b = [r["f1"] for r in runs if "f1" in r]
            if len(a) >= 2 and len(b) >= 2:
                t = paired_test(a, b)
                row["p (F1)"] = (f"{t['t_p']:.3f}"
                                 if np.isfinite(t["t_p"]) else "—")
                csv_row["p_f1_vs_ref"]   = t["t_p"]
                csv_row["cohens_dz_f1"]  = t["cohens_dz"]
            else:
                row["p (F1)"] = "—"
        else:
            row["p (F1)"] = "ref"

        rows.append(row)
        csv_rows.append(csv_row)

    columns = ["Method"] + [HEADERS[m] for m in METRICS] + ["p (F1)"]

    _write_csv(os.path.join(RESULTS_DIR, f"{prefix}.csv"), csv_rows,
               fieldnames=sorted({k for r in csv_rows for k in r}))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.tex"),
                _latex_table(rows, columns,
                             "Comparison against baselines. Values are mean "
                             r"$\pm$ std over seeds. $^{\dagger}$ denotes methods "
                             "with no privacy guarantee (raw data pooled); these "
                             "are utility upper bounds, not competing systems. "
                             "$p$ from a paired $t$-test on F1 against the "
                             "proposed method.",
                             "tab:comparison"))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.md"), _md_table(rows, columns))
    return rows


# ── Table 2: ablations ────────────────────────────────────────────────────────
def report_ablations(ablation_results, reference="full",
                     prefix="table2_ablation"):
    """
    Emits the ablation table with the DELTA against the full system, which is
    the column a reader actually reads.
    """
    rows, csv_rows = [], []
    ref_runs = ablation_results.get(reference, [])
    ref_agg  = aggregate_runs(ref_runs, METRICS) if ref_runs else {}

    for variant, runs in ablation_results.items():
        if not runs:
            continue
        agg   = aggregate_runs(runs, METRICS)
        label = runs[0].get("label", variant)
        dagger = r"$^{\dagger}$" if variant in NO_PRIVACY else ""

        row = {"Variant": label + dagger}
        csv_row = {"variant": variant, "label": label, "n_seeds": len(runs)}

        for m in METRICS:
            if m in agg:
                row[HEADERS[m]] = fmt(agg[m])
                row[f"_{HEADERS[m]}_mean"] = agg[m]["mean"]
                csv_row[f"{m}_mean"] = round(agg[m]["mean"], 6)
                csv_row[f"{m}_std"]  = round(agg[m]["std"], 6)

        if variant != reference and "f1" in agg and "f1" in ref_agg:
            d = agg["f1"]["mean"] - ref_agg["f1"]["mean"]
            row[r"$\Delta$F1"] = f"{d:+.4f}"
            csv_row["delta_f1"] = round(d, 6)
            a = [r["f1"] for r in ref_runs if "f1" in r]
            b = [r["f1"] for r in runs if "f1" in r]
            if len(a) >= 2 and len(b) >= 2:
                csv_row["p_f1_vs_full"] = paired_test(a, b)["t_p"]
        else:
            row[r"$\Delta$F1"] = "ref"

        rows.append(row)
        csv_rows.append(csv_row)

    columns = ["Variant"] + [HEADERS[m] for m in METRICS] + [r"$\Delta$F1"]

    _write_csv(os.path.join(RESULTS_DIR, f"{prefix}.csv"), csv_rows,
               fieldnames=sorted({k for r in csv_rows for k in r}))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.tex"),
                _latex_table(rows, columns,
                             "Component ablation. Each variant removes exactly "
                             "one component from the full system. "
                             r"$\Delta$F1 is relative to the full system.",
                             "tab:ablation"))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.md"), _md_table(rows, columns))
    return rows


# ── Table 3: scaling ──────────────────────────────────────────────────────────
def report_scaling(scaling_results, prefix="table3_scaling"):
    rows, csv_rows = [], []

    for k in sorted(scaling_results):
        runs = scaling_results[k]
        if not runs:
            continue
        agg = aggregate_runs(runs, METRICS + ["sec_per_round", "mean_sigma",
                                              "min_client_size"])
        row = {"Banks": str(k)}
        csv_row = {"num_clients": k, "n_seeds": len(runs)}

        for m in METRICS:
            if m in agg:
                row[HEADERS[m]] = fmt(agg[m])
                row[f"_{HEADERS[m]}_mean"] = agg[m]["mean"]
                csv_row[f"{m}_mean"] = round(agg[m]["mean"], 6)
                csv_row[f"{m}_std"]  = round(agg[m]["std"], 6)

        if "mean_sigma" in agg:
            row[r"$\bar{\sigma}$"] = f"{agg['mean_sigma']['mean']:.3f}"
            csv_row["mean_sigma"]  = round(agg["mean_sigma"]["mean"], 6)
        if "sec_per_round" in agg:
            row["s/round"] = f"{agg['sec_per_round']['mean']:.1f}"
            csv_row["sec_per_round"] = round(agg["sec_per_round"]["mean"], 3)

        km = int(np.mean([r.get("kmeans_smote_used", 0) for r in runs]))
        row["KMeansSMOTE"] = f"{km}/{k}"
        csv_row["kmeans_smote_used"] = km
        csv_row["min_client_size"] = int(np.mean(
            [r.get("min_client_size", 0) for r in runs]))

        rows.append(row)
        csv_rows.append(csv_row)

    columns = ["Banks"] + [HEADERS[m] for m in METRICS] + \
              [r"$\bar{\sigma}$", "s/round", "KMeansSMOTE"]

    _write_csv(os.path.join(RESULTS_DIR, f"{prefix}.csv"), csv_rows,
               fieldnames=sorted({k for r in csv_rows for k in r}))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.tex"),
                _latex_table(rows, columns,
                             "Scalability across federation size. Total data is "
                             "held fixed, so per-bank data shrinks as banks are "
                             r"added and per-bank DP noise $\bar{\sigma}$ rises. "
                             "The KMeansSMOTE column reports how many banks could "
                             "actually support cluster-based oversampling.",
                             "tab:scaling"))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.md"), _md_table(rows, columns))
    return rows


# ── Table 4: privacy budget ───────────────────────────────────────────────────
def report_privacy(privacy_plan, epsilon_spent=None, prefix="table4_privacy"):
    rows, csv_rows = [], []
    for i, r in enumerate(privacy_plan):
        spent = (epsilon_spent[i] if epsilon_spent and i < len(epsilon_spent)
                 else float("nan"))
        rows.append({
            "Bank":            str(r["client_id"]),
            "$n_k$":           f"{r['n_samples']:,}",
            "$q_k$":           f"{r['sample_rate']:.5f}",
            "Steps":           str(r["steps"]),
            r"$\sigma_k$":     f"{r['sigma']:.4f}",
            r"$\delta_k$":     f"{r['delta']:.1e}",
            r"$\varepsilon$ spent": f"{spent:.3f}" if spent == spent else "—",
        })
        csv_rows.append({
            "client_id": r["client_id"], "n_samples": r["n_samples"],
            "sample_rate": r["sample_rate"], "steps": r["steps"],
            "sigma": r["sigma"], "delta": r["delta"],
            "epsilon_target": r["epsilon"], "epsilon_spent": spent,
        })

    columns = ["Bank", "$n_k$", "$q_k$", "Steps", r"$\sigma_k$",
               r"$\delta_k$", r"$\varepsilon$ spent"]

    _write_csv(os.path.join(RESULTS_DIR, f"{prefix}.csv"), csv_rows,
               fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.tex"),
                _latex_table(rows, columns,
                             "Per-bank DP-SGD calibration. Every bank is tuned to "
                             "a common target $\\varepsilon$; smaller banks "
                             "receive proportionally more noise because each "
                             "record participates in a larger fraction of batches.",
                             "tab:privacy", bold_best=False))
    _write_text(os.path.join(RESULTS_DIR, f"{prefix}.md"), _md_table(rows, columns))
    return rows


# ── Figures ───────────────────────────────────────────────────────────────────
# White background throughout (not the earlier dark theme) — these figures go
# straight into the paper/report, which are printed on white, not viewed on
# screen against a dark dashboard.
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color="#1a1d27", fontsize=13)
    ax.set_xlabel(xlabel, color="#1a1d27", fontsize=11)
    ax.set_ylabel(ylabel, color="#1a1d27", fontsize=11)
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#1a1d27")
    ax.spines[:].set_color("#888")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)


def plot_comparison_bars(results_by_method, labels=None,
                         filename="comparison_baselines.png"):
    """Grouped bars with std error bars across methods."""
    labels = labels or {}
    methods = [m for m, r in results_by_method.items() if r]
    if not methods:
        return None

    show = ["recall", "f1", "pr_auc"]
    x = np.arange(len(show))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#ffffff")
    cmap = plt.get_cmap("tab10")

    for i, m in enumerate(methods):
        agg  = aggregate_runs(results_by_method[m], show)
        vals = [agg.get(k, {}).get("mean", 0) for k in show]
        errs = [agg.get(k, {}).get("std", 0) for k in show]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs,
               capsize=3, label=labels.get(m, m), color=cmap(i % 10),
               edgecolor="#1a1d27", linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([HEADERS[k] for k in show], color="#1a1d27", fontsize=12)
    ax.set_ylim(0, 1.05)
    _style(ax, "Proposed system vs baselines (mean ± std over seeds)", "", "Score")
    ax.legend(facecolor="#ffffff", edgecolor="#888", labelcolor="#1a1d27", fontsize=9, ncol=2)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_ablation_deltas(ablation_results, reference="full",
                         filename="ablation_deltas.png"):
    """Horizontal bars of ΔF1 vs the full system — negative means the component helps."""
    ref = aggregate_runs(ablation_results.get(reference, []), ["f1"])
    if "f1" not in ref:
        return None

    items = []
    for v, runs in ablation_results.items():
        if v == reference or not runs:
            continue
        agg = aggregate_runs(runs, ["f1"])
        if "f1" in agg:
            items.append((runs[0].get("label", v),
                          agg["f1"]["mean"] - ref["f1"]["mean"],
                          agg["f1"]["std"]))
    if not items:
        return None
    items.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(items) + 3))
    fig.patch.set_facecolor("#ffffff")
    ys     = np.arange(len(items))
    deltas = [i[1] for i in items]
    errs   = [i[2] for i in items]
    colors = ["#DC2626" if d < 0 else "#059669" for d in deltas]

    ax.barh(ys, deltas, xerr=errs, capsize=3, color=colors,
            edgecolor="#1a1d27", linewidth=0.6, zorder=3)
    ax.axvline(0, color="#1a1d27", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], color="#1a1d27", fontsize=10)
    _style(ax, "Ablation: ΔF1 when each component is removed", "ΔF1 vs full system", "")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_scaling_curves(scaling_results, filename="scaling_curves.png"):
    ks = sorted(k for k, v in scaling_results.items() if v)
    if not ks:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#ffffff")

    panels = [(["recall", "f1", "pr_auc"], "Detection quality", "Score"),
              (["mean_sigma"],             "Mean per-bank DP noise", r"$\bar{\sigma}$"),
              (["sec_per_round"],          "Wall-clock cost", "Seconds / round")]
    cmap = plt.get_cmap("tab10")

    for ax, (keys, title, ylab) in zip(axes, panels):
        for i, key in enumerate(keys):
            means = [aggregate_runs(scaling_results[k], [key]).get(key, {}).get("mean", 0) for k in ks]
            stds  = [aggregate_runs(scaling_results[k], [key]).get(key, {}).get("std", 0) for k in ks]
            ax.errorbar(ks, means, yerr=stds, marker="o", markersize=6,
                        linewidth=2, capsize=4, color=cmap(i % 10),
                        label=HEADERS.get(key, key))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        _style(ax, title, "Number of banks", ylab)
        ax.legend(facecolor="#ffffff", edgecolor="#888", labelcolor="#1a1d27", fontsize=9)

    plt.suptitle("Scalability across federation size", color="#1a1d27",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_summary_markdown(sections, path=None):
    """Assemble every emitted .md fragment into one readable results file."""
    path = path or os.path.join(RESULTS_DIR, "RESULTS.md")
    parts = ["# EBA-FFD v2 — Experimental Results\n"]
    for title, body in sections:
        parts.append(f"\n## {title}\n\n{body}\n")
    parts.append(
        "\n---\n\n**†** marks methods with no privacy guarantee (raw data pooled "
        "or privacy ignored). These are utility upper bounds that quantify the "
        "cost of privacy, not competing systems. FedProx at matched ε is the "
        "like-for-like federated comparison.\n")
    return _write_text(path, "".join(parts))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake = lambda mu: [{"precision": rng.normal(mu, .01), "recall": rng.normal(mu, .01),
                        "f1": rng.normal(mu, .01), "auc": rng.normal(.97, .005),
                        "pr_auc": rng.normal(mu, .01)} for _ in range(3)]
    res = {"eba_ffd": fake(.80), "xgboost": fake(.84), "fedprox": fake(.76)}
    report_comparison(res, labels={"eba_ffd": "EBA-FFD (ours)",
                                   "xgboost": "XGBoost", "fedprox": "FedProx"})
    plot_comparison_bars(res)
    print("[reporting] wrote:", sorted(os.listdir(RESULTS_DIR)))
