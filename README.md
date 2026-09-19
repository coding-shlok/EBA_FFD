# EBA-FFD — Adaptive Federated Fraud Detection under Differential Privacy and Concept Drift

A federated fraud-detection system for banks that legally cannot pool
transaction data. Each bank trains locally under a **formal differential
privacy guarantee**; the server aggregates with a **drift-aware trust
weighting** that detects and damps a bank whose local data quality degrades
mid-training, instead of silently trusting it or excluding it outright.

Full results, methodology, and a paper draft: see
[`EBA-FFD_ICECCE2026_Paper.docx`](EBA-FFD_ICECCE2026_Paper.docx) and
[`EBA-FFD_Capstone_Project_Report.docx`](EBA-FFD_Capstone_Project_Report.docx).

---

## Headline results (full-scale GPU run — 100% data, 3 seeds, 4–32 banks)

| Method | F1 | vs. EBA-FFD |
|---|---|---|
| **EBA-FFD (ours, DP, ε = 3.0)** | **0.79** | reference |
| XGBoost (centralized, no privacy) | 0.83 | significant, *p* = 0.012 |
| Local-only (no collaboration) | 0.77 | significant, *p* = 0.034 |
| FedProx + DP (like-for-like federated baseline) | 0.80 | not significant, *p* = 0.423 |

**Under injected concept drift** (a bank's labels corrupted mid-training):

| | F1 (stationary) | F1 (under drift) |
|---|---|---|
| EBA-FFD, adaptive aggregation | 0.79 | **0.66** |
| Vanilla FedAvg (no adaptation) | 0.79 | **0.23** (collapse) |

Federated collaboration under privacy has a real, statistically significant
cost against a non-private centralized model — and a real, statistically
significant benefit over any single bank acting alone. Adaptive aggregation
does not make the system immune to a corrupted bank, but it substantially
contains the damage where standard FedAvg does not.

---

## The story behind these numbers

Three things changed between an early pilot evaluation (20% data, 2 seeds)
and the final full-scale run (100% data, 3 seeds, GPU), each one a real
finding worth knowing about before trusting any small-scale FL evaluation:

1. **A "no significant difference" result at pilot scale was wrong.** The
   pilot run showed EBA-FFD statistically indistinguishable from XGBoost.
   At full scale, with more statistical power, the same underlying gap
   became clearly significant (*p* = 0.012). The gap was real all along —
   the pilot just lacked the power to detect it.
2. **The originally-designed architecture wasn't the best one.** A capacity
   sweep under DP-SGD showed a plain MLP consistently beating both a
   CNN-only and a CNN+BiLSTM classifier, confirmed at three seeds and again
   at full scale. Fewer parameters means less surface for DP-SGD's
   per-sample noise to corrupt.
3. **Focal Loss, added specifically to help class imbalance, was net
   harmful.** At full data volume, KMeans-SMOTE/ENN balancing alone already
   handled the imbalance Focal Loss was meant to fix; removing it gave the
   single largest gain in the whole ablation study (+0.02 F1). Plain BCE is
   now the default.

Two measurement bugs also had to be found and fixed before the drift result
was even visible:

- **Best-round selection was silently broken.** Selecting the "best"
  training round by validation F1 ties across rounds (the decision
  threshold is re-optimized against validation every round), so a naive
  `max()` always fell back to the first round tried. Fixed by selecting on
  validation PR-AUC instead, which has no threshold to re-optimize against.
- **Drift injection as originally designed was a structural no-op.**
  Covariate-shift injection was neutralized by the model's normalization
  layer before it ever reached the loss (switched to label-flip injection),
  and a fixed drift-target client index could land on a near-empty bank
  under some random partitions (switched to always target the largest
  bank). Worse: any validation-based "best round" selection under drift
  structurally favors the pre-drift round for every method alike, since
  injected corruption can only hurt — erasing the entire comparison the
  experiment was designed to make. Fixed by reporting the **final round**
  under the drift condition instead of a best-round snapshot.

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# place creditcard.csv at data/creditcard.csv
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# 1. Validate the pipeline end to end (~10 min, numbers NOT meaningful)
python run_all_experiments.py --preset smoke

# 2. Sanity-check trends (~1-3 h)
python run_all_experiments.py --preset quick

# 3. Produce the reportable numbers (many hours — use a GPU)
python run_all_experiments.py --preset full
```

Single run instead of the full suite:

```bash
python main.py --rounds 10 --clients 4 --epsilon 3.0
python main.py --drift                    # demonstrate adaptive aggregation
python main.py --no-dp                    # privacy-free upper bound
python main.py --aggregation fedavg       # vanilla FedAvg control
```

Outputs: tables in `results/` (`.csv`, `.tex`, `.md`), figures in
`outputs/plots/`, summary in `results/RESULTS.md`.

**Running on a GPU cluster (Altair Access):** see
[`ALTAIR_SUBMISSION_GUIDE.md`](ALTAIR_SUBMISSION_GUIDE.md) and
[`run_eba_ffd_altair.ipynb`](run_eba_ffd_altair.ipynb) for the job-submission
notebook this project's full-scale results were actually produced with.

---

## System design

1. **Non-IID partitioning** — Dirichlet(α = 0.5) split across simulated
   banks, with a minimum fraud-per-client floor.
2. **Local class balancing** — KMeans-SMOTE, falling back to plain SMOTE for
   clients whose cluster structure is too small, followed by ENN cleaning.
   Never leaves the client; doesn't touch the DP budget.
3. **Right-sized classifier** — a plain MLP with BCE loss (see "The story
   behind these numbers" above for why, not CNN/BiLSTM/Focal Loss as
   originally designed).
4. **Differentially private local training** — DP-SGD via Opacus, per-sample
   gradient clipping, an RDP accountant composing (ε, δ) across every round,
   individually calibrated per bank so different dataset sizes reach the
   same shared privacy budget.
5. **Drift-aware adaptive aggregation** — a Page-Hinkley test on each bank's
   local training loss detects drift; a drifting bank's aggregation weight
   is damped (never silenced — floor 0.20) rather than excluded.

```
severity_k = PH_k / lambda
trust_k    = clip(exp(-beta * severity_k), trust_min, 1)
p_k        = (n_k * trust_k) / sum_j (n_j * trust_j)
```

Reduces exactly to vanilla FedAvg when no drift is detected — under
stationary conditions `full ≈ no_adaptive` is the *expected* result, not a
failure; the mechanism only shows its effect under the drift stage
(`--only drift`, `drift_enabled=True`).

---

## Framing the baseline table honestly

XGBoost, Logistic Regression, centralized CNN, and local-only carry **no
privacy guarantee** — they pool raw data or ignore privacy entirely. These
are **utility upper bounds that quantify the price of privacy, not competing
systems**, marked with † in every generated table. If XGBoost beats EBA-FFD,
that's expected — gradient-boosted trees usually win on tabular data, and
that's the cost of the privacy guarantee, not a failure of the method.
**FedProx at matched ε is the only genuinely like-for-like comparison.**

---

## Per-bank privacy calibration

Banks hold very different amounts of data. Under one fixed noise multiplier
they'd get wildly different guarantees, so σ is solved per bank from its own
sampling rate and step count so every bank reaches the **same** ε:

```
sigma_k = argmin { sigma : RDP(sigma, q_k, T_k) <= epsilon_target }
```

Full-scale calibration (ε = 3.0) — smaller banks correctly receive more noise:

| Bank | n_k | σ_k |
|---|---|---|
| 3 | 218,467 | 0.69 |
| 1 | 23,789 | 1.11 |
| 2 | 13,956 | 1.35 |
| 0 | 2,212 | 3.03 |

---

## Known limitations

1. **Single dataset.** ULB creditcard only. The systems-level findings
   (non-IID sensitivity, DP-SGD's differential effect across architectures,
   the value of trust-weighted aggregation) aren't dataset-specific claims,
   but haven't been checked against a second dataset.
2. **n = 3 seeds.** The minimum for a non-degenerate paired Wilcoxon test.
   The drift-condition result in particular carries very high seed-to-seed
   variance (std 0.19–0.53), traced to how dominant the drift-targeted
   (largest) bank happens to be under a given random partition. More seeds
   would sharpen this considerably.
3. **Simulated, non-adversarial drift.** Drift is injected as an accidental
   label-flip corruption, not an adversary deliberately crafting updates to
   evade the Page-Hinkley detector — a stronger, distinct threat model
   worth testing separately.
4. **32-bank results are data-thin.** ~350 training fraud records over 32
   banks is ~11 each; degradation there mixes "the method scales poorly"
   with "there is almost no fraud data per bank."

---

## Module map

```
config.py                 single source of truth; merged overrides
data_loader.py            non-IID partitioning, min-fraud floor, val split
preprocessing.py          balancing + honest fallback logging
model.py                  MLP/CNN/BiLSTM with ablation + DP-compatible switches
drift.py                  Page-Hinkley detector, trust scoring, drift injection
privacy.py                DP-SGD calibration, RDP accounting
client.py                 local DP-SGD training, FedProx proximal term
server.py                 adaptive aggregation, threshold calibration, plots
training_loop.py          single-run orchestrator
evaluation.py             shared metrics + threshold selection
stats_utils.py            multi-seed aggregation, paired tests
baselines.py              XGBoost / LogReg / FedProx / FedAvg / local-only
tree_baselines.py         XGBoost run in a torch-free subprocess (avoids an
                           OpenMP conflict between PyTorch and XGBoost on macOS)
ablations.py              8 single-component ablations (architecture, loss,
                           balancing, DP, aggregation)
scaling.py                4/8/16/32-bank sweep
reporting.py              CSV / LaTeX / Markdown tables + figures (white background)
run_all_experiments.py    checkpointed unattended runner, 5 stages
run_eba_ffd_altair.ipynb  thin job notebook for GPU cluster submission
explainability.py         SHAP + LIME (not currently wired into the main pipeline)
main.py                   single-run entry point
```

---

## Runtime

DP-SGD is roughly an order of magnitude slower than ordinary training
(per-sample gradients are materialized for every parameter). The `full`
preset is measured in many GPU-hours; every stage is checkpointed to
`results/checkpoints/`, so a crash or restart doesn't cost completed work.
Results in this README were produced on an NVIDIA H100 (via Altair Access).

---

## References

Full IEEE-formatted bibliography (26 entries) is in the paper draft. Key ones:

1. McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)
2. Abadi et al. (2016) — Deep Learning with Differential Privacy (DP-SGD)
3. Li et al. (2020) — Federated Optimization in Heterogeneous Networks (FedProx)
4. Page (1954) — Continuous Inspection Schemes (Page-Hinkley)
5. Mironov (2017) — Rényi Differential Privacy
6. Lin et al. (2017) — Focal Loss for Dense Object Detection
7. Chawla et al. (2002) — SMOTE
8. Last et al. (2017) — KMeans-SMOTE oversampling
9. Yousefpour et al. (2021) — Opacus: User-Friendly Differential Privacy Library in PyTorch
10. Dal Pozzolo et al. (2015) — ULB dataset / undersampling probability calibration
