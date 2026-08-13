# EBA-FFD v2 — Explainable, Balanced, Adaptive Federated Fraud Detection

Federated credit-card fraud detection with drift-aware aggregation, formal
differential privacy, and a full experimental protocol for publication.

This is v1 (the competition system) upgraded along seven axes for a paper
submission. Everything in v1 still works; the additions are described below.

---

## What changed from v1

| # | Upgrade | Where it lives |
|---|---------|----------------|
| 1 | **Adaptive FedAvg** — Page-Hinkley drift detection per bank, trust-weighted aggregation | `drift.py`, `server.py` |
| 2 | **Real differential privacy** — Opacus DP-SGD, RDP accountant, per-bank σ calibration | `privacy.py`, `client.py` |
| 3 | **Comparison baselines** — XGBoost, Logistic Regression, FedProx, FedAvg, local-only, centralized NN | `baselines.py` |
| 4 | **Ablation study** — 8 variants, one component removed at a time | `ablations.py` |
| 5 | **Statistical rigour** — multi-seed mean ± std, paired significance tests | `stats_utils.py` |
| 6 | **Scalability study** — 4 / 8 / 16 / 32 banks | `scaling.py` |
| 7 | **Automated runner** — one command, all tables and figures, checkpointed | `run_all_experiments.py`, `reporting.py` |

Plus two correctness fixes and one methodological fix described under
**Issues found in v1** below. Read that section before writing your methods.

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

# 3. Produce the paper (many hours — use a GPU)
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

---

## Issues found in v1 — read before writing your paper

### 1. KMeans-SMOTE never actually ran

`preprocessing.py` defined `apply_kmeans_smoteenn` **twice**. The first
definition was truncated mid-body; Python silently discarded it and kept the
second, which called plain `SMOTE`. The README, the module docstring and the
system diagram all claimed KMeans-SMOTE. **Every v1 number was produced by
vanilla SMOTE + ENN.**

Worse: when the real `KMeansSMOTE` is wired in, it **fails on most clients** —
"no clusters found with sufficient samples of class 1". With 492 fraud records
Dirichlet-split across banks, there are too few minority samples per client to
form valid clusters. This is a property of the dataset, not a bug you can fix.

v2 attempts KMeansSMOTE, falls back explicitly, and **logs which method each
client actually used**. Quote those counts in the paper rather than claiming a
method that did not run. At 16 and 32 banks, expect KMeansSMOTE on close to
zero clients.

### 2. v1's "differential privacy" had no privacy guarantee

v1 clipped *gradients* during training, then added Gaussian noise to the
*weights* afterwards. Those operations are unconnected: clipping a minibatch
gradient does not bound the sensitivity of the resulting weight vector to a
single record, so no (ε, δ) statement follows. It was a regulariser, not DP.

v2 uses genuine DP-SGD via Opacus: per-sample clipping, noise on summed clipped
gradients, Poisson subsampling, RDP accounting composed across all rounds.

This forced two architecture changes, both automatic when `dp_mode="opacus"`:
`BatchNorm1d → GroupNorm` (BatchNorm couples samples, so per-sample gradients
are undefined) and `nn.LSTM → opacus.layers.DPLSTM`.

### 3. Recall at a fixed 0.5 threshold is meaningless under DP

DP noise compresses predicted probabilities toward the base rate. In testing,
a model with **ROC-AUC 0.999 scored recall 0.000** — near-perfect ranking, but
nothing crossed 0.5.

v2 adds a **validation split** and calibrates the decision threshold on it,
never on test. The same calibration is applied to every baseline, so no method
gets a tuned threshold while others use a fixed one.

**Report PR-AUC as your headline metric.** At a 0.17 % fraud rate, ROC-AUC is
optimistic and compresses differences — 0.96 sounds far better than it is.

---

## Framing the baseline table honestly

XGBoost, Logistic Regression, centralized CNN-BiLSTM and local-only carry **no
privacy guarantee** — they pool raw data or ignore privacy. Your system runs at
finite ε. These are **utility upper bounds that quantify the price of privacy,
not competing systems.** They are marked with † in every generated table.

If XGBoost beats you, that is expected — gradient-boosted trees usually win on
tabular data. That is not a failure of your method; it is the cost of the
guarantee. State it that way.

**FedProx at matched ε is your only genuinely like-for-like comparison.** Lead
with it.

---

## Contribution 1: how Adaptive FedAvg works

Classical FedAvg weights client *k* by *n_k / N*, fixed forever. That is optimal
only when every client's distribution is stationary.

v2 modulates each weight by a trust score from a Page-Hinkley test on that
client's mean local loss:

```
severity_k = PH_k / lambda
trust_k    = clip(exp(-beta * severity_k), trust_min, 1)
p_k        = (n_k * trust_k) / sum_j (n_j * trust_j)
```

Properties worth stating:

- **Reduces exactly to FedAvg when no drift is detected**, so it is a strict
  generalisation.
- `trust_min > 0` means a drifting bank is **damped, never silenced** — after a
  genuine regime change that bank holds the only evidence about the new fraud
  pattern.
- Needs **one scalar per client per round**. The server sees no features and no
  extra gradients, so drift detection costs **zero additional privacy budget**.

**Evaluating it requires drift to exist.** Under stationary conditions
`full ≈ no_adaptive` is the *expected* result, not a failure. That is why
`run_all_experiments.py` runs a separate drift stage (`--only drift`) with
`drift_enabled=True`. The headline figure is
`outputs/plots/adaptive_trust_dynamics.png`: the drifting client's trust should
collapse right after injection and recover afterwards.

---

## Contribution 2: per-bank privacy calibration

Banks hold different amounts of data. Under a single fixed noise multiplier
they would receive wildly different guarantees. v2 inverts this — every bank
gets the **same ε**, and σ is solved per bank from its own sampling rate and
step count:

```
sigma_k = argmin { sigma : RDP(sigma, q_k, T_k) <= epsilon_target }
```

Smaller banks receive **more** noise, which is the correct direction. Verified
calibration (ε = 3.0):

| n_k | σ_k |
|---|---|
| 50,000 | 0.897 |
| 20,000 | 1.184 |
| 5,000 | 2.070 |
| 1,200 | 4.082 |

---

## Known limitations to disclose

1. **Single dataset.** ULB creditcard only. Reviewers will ask for a second
   (IEEE-CIS, PaySim). The strongest single addition you could make.
2. **32-bank results are confounded.** ~350 training fraud records over 32
   banks is ~11 each. Degradation there mixes "the method scales poorly" with
   "there is almost no fraud data per bank". State this explicitly or a
   reviewer will read the curve as a weakness of your method.
3. **n = 3 seeds.** Wilcoxon cannot go below p = 0.25 at n = 3, so it is
   reported for completeness only; the paired t-test is the usable test and
   rests on a normality assumption 3 points cannot verify. If pushed, the fix
   is more seeds (n ≥ 10), not a different test.
4. **Simulated drift.** Concept drift is injected synthetically. Real temporal
   drift in the ULB data is not separately validated.
5. **The LSTM processes feature positions, not transaction sequences.** The
   dataset has no account identifiers, so there are no true per-account
   sequences. Describe the recurrent block accurately — do not imply temporal
   modelling the data cannot support.

---

## Module map

```
config.py                 single source of truth; merged overrides
data_loader.py            non-IID partitioning, min-fraud floor, val split
preprocessing.py          balancing + honest fallback logging
model.py                  CNN/BiLSTM with ablation + DP-compatible switches
drift.py                  Page-Hinkley detector, trust scoring, drift injection
privacy.py                DP-SGD calibration, RDP accounting
client.py                 local DP-SGD training, FedProx proximal term
server.py                 adaptive aggregation, threshold calibration, plots
training_loop.py          single-run orchestrator
evaluation.py             shared metrics + threshold selection
stats_utils.py            multi-seed aggregation, paired tests
baselines.py              XGBoost / LogReg / FedProx / FedAvg / local-only
ablations.py              8 single-component ablations
scaling.py                4/8/16/32-bank sweep
reporting.py              CSV / LaTeX / Markdown tables + figures
run_all_experiments.py    checkpointed unattended runner
explainability.py         SHAP + LIME
main.py                   single-run entry point
```

---

## Runtime warning

DP-SGD with a DPLSTM is roughly an order of magnitude slower than ordinary
training: per-sample gradients are materialised for every parameter and the
recurrent block cannot use cuDNN's fused kernels. Measured ~25–70 s per round
on a single CPU core at a small subsample. **Use a GPU for the `full` preset,
and run `--only scaling` last.** Every stage is checkpointed to
`results/checkpoints/`, so a crash does not cost completed work.

---

## References

1. McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)
2. Abadi et al. (2016) — Deep Learning with Differential Privacy (DP-SGD)
3. Li et al. (2020) — Federated Optimization in Heterogeneous Networks (FedProx)
4. Page (1954) — Continuous Inspection Schemes (Page-Hinkley)
5. Mouss et al. (2004) — Test of Page-Hinkley for fault detection
6. Lin et al. (2017) — Focal Loss for Dense Object Detection
7. Chawla et al. (2002) — SMOTE
8. Last et al. (2017) — KMeans-SMOTE oversampling
9. Lundberg & Lee (2017) — SHAP
10. Ribeiro et al. (2016) — LIME
11. Yousefpour et al. (2021) — Opacus: User-Friendly Differential Privacy Library in PyTorch
