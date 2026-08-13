"""
drift.py
========
Concept-drift detection for Adaptive Federated Averaging  (contribution #1).

Motivation
──────────
Classical FedAvg weights client k by n_k / N and nothing else. That is optimal
only when every client's local distribution stays fixed. In fraud detection it
does not: a bank onboards a new merchant category, a fraud ring switches
tactics, a payment corridor changes. The affected bank's local optimum drifts
away from the federation's, and its update starts dragging the global model
toward a stale or adversarially-shifted objective.

Detector — Page-Hinkley test (Page, 1954; Mouss et al., 2004)
─────────────────────────────────────────────────────────────
The server observes one scalar per client per round: that client's mean local
training loss. Under a stable distribution this sequence is roughly stationary
or slowly decreasing. Drift shows up as a sustained *increase*.

Page-Hinkley tracks the cumulative deviation of the stream from its own running
mean, minus a tolerance delta:

    x_bar_t = x_bar_{t-1} + (x_t - x_bar_{t-1}) / t
    m_t     = alpha * m_{t-1} + (x_t - x_bar_t - delta)
    M_t     = min(m_1, ..., m_t)
    PH_t    = m_t - M_t

PH_t stays near zero while the stream is stationary, and grows once the mean
shifts upward. An alarm fires when PH_t > lambda.

  delta     tolerance: changes smaller than this are treated as noise
  lambda    alarm threshold: larger = fewer false alarms, slower detection
  alpha     forgetting factor, discounts stale evidence

Why Page-Hinkley and not a windowed distribution test: it is single-pass, O(1)
in memory and time, and needs only a scalar the client already reports. The
server never sees client features, so drift detection costs zero extra privacy
budget — an important property given contribution #2.

From detection to aggregation weight
────────────────────────────────────
A hard alarm bit would make aggregation weights jump discontinuously between
rounds. Instead we convert the raw PH statistic into a bounded, continuous
trust score:

    severity_k = PH_k / lambda                     (0 = stable, >1 = alarming)
    trust_k    = clip(exp(-beta * severity_k), trust_min, 1)

and the server aggregates with

    p_k = (n_k * trust_k) / sum_j (n_j * trust_j)

trust_min > 0 is deliberate: a drifting bank is damped, never silenced. Its
data is still the only evidence the federation has about the new regime, and
zeroing it out would prevent the global model from ever adapting.
"""

import numpy as np


class PageHinkley:
    """
    One-sided Page-Hinkley change detector (detects increases in the mean).

    Usage:
        ph = PageHinkley(delta=0.005, threshold=0.05)
        result = ph.update(loss_value)
        result["detected"]  -> bool
        result["ph_stat"]   -> float, the PH_t statistic
        result["severity"]  -> float, PH_t / lambda
    """

    def __init__(self, delta=0.005, threshold=0.05, alpha=0.9999, min_rounds=3):
        self.delta      = delta
        self.threshold  = threshold
        self.alpha      = alpha
        self.min_rounds = min_rounds
        self.reset()

    def reset(self):
        self.n         = 0
        self.x_mean    = 0.0
        self.m_t       = 0.0     # cumulative deviation
        self.M_t       = 0.0     # running minimum of m_t
        self.ph_stat   = 0.0
        self.detected  = False

    def update(self, x):
        """Feed one observation (this round's mean local loss) into the test."""
        x = float(x)
        self.n += 1

        # Running mean of the stream
        self.x_mean += (x - self.x_mean) / self.n

        # Cumulative deviation above the tolerated drift magnitude
        self.m_t = self.alpha * self.m_t + (x - self.x_mean - self.delta)
        self.M_t = min(self.M_t, self.m_t)

        self.ph_stat = self.m_t - self.M_t
        self.detected = (self.n >= self.min_rounds) and (self.ph_stat > self.threshold)

        return {
            "detected": self.detected,
            "ph_stat":  self.ph_stat,
            "severity": self.ph_stat / self.threshold if self.threshold > 0 else 0.0,
            "mean":     self.x_mean,
            "n":        self.n,
        }


class ClientDriftMonitor:
    """
    Wraps a PageHinkley detector for a single client and converts its output
    into an aggregation trust score with hysteresis.

    Hysteresis (trust_recovery) prevents a bank from snapping straight back to
    full trust the round after an alarm clears: trust climbs back gradually.
    """

    def __init__(self, client_id, detector_cfg=None, beta=2.0,
                 trust_min=0.2, recovery=0.3):
        detector_cfg = detector_cfg or {}
        self.client_id = client_id
        self.detector  = PageHinkley(
            delta      = detector_cfg.get("delta", 0.005),
            threshold  = detector_cfg.get("threshold", 0.05),
            alpha      = detector_cfg.get("alpha", 0.9999),
            min_rounds = detector_cfg.get("min_rounds", 3),
        )
        self.beta      = beta
        self.trust_min = trust_min
        self.recovery  = recovery

        self.trust        = 1.0
        self.history      = []   # one record per round
        self.n_alarms     = 0
        self.alarm_rounds = []

    def update(self, loss, round_num=None):
        """
        Feed this round's mean local loss, update trust, return a record.
        """
        res = self.detector.update(loss)

        # Instantaneous trust implied by the current drift severity
        target_trust = float(np.clip(
            np.exp(-self.beta * max(0.0, res["severity"])),
            self.trust_min, 1.0
        ))

        if target_trust < self.trust:
            # Drift detected: drop immediately. Reacting fast is the whole point.
            self.trust = target_trust
        else:
            # Recovering: ease back up so weights don't oscillate round to round.
            self.trust += self.recovery * (target_trust - self.trust)

        if res["detected"]:
            self.n_alarms += 1
            if round_num is not None:
                self.alarm_rounds.append(round_num)

        record = {
            "round":     round_num,
            "client_id": self.client_id,
            "loss":      float(loss),
            "ph_stat":   float(res["ph_stat"]),
            "severity":  float(res["severity"]),
            "detected":  bool(res["detected"]),
            "trust":     float(self.trust),
        }
        self.history.append(record)
        return record


# ── Drift injection (used to VALIDATE the detector, not part of the method) ────
def inject_drift(client, mode="covariate_shift", level=0.5, seed=None):
    """
    Deliberately corrupt one client's local data to simulate a distribution
    change. This exists so the paper can show adaptive aggregation actually
    helps when drift occurs — it is an evaluation harness, not a component
    of the proposed system.

    Modes:
      covariate_shift : add a constant offset + noise to features (new merchant
                        mix, new payment corridor)
      scale           : multiplicatively rescale features (currency/limit change)
      label_flip      : flip a fraction of labels (fraud ring changes tactics so
                        old labels no longer describe the new concept)
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    rng = np.random.default_rng(seed)
    dataset = client.dataloader.dataset
    X = dataset.tensors[0].numpy().copy()
    y = dataset.tensors[1].numpy().copy()

    if mode == "covariate_shift":
        shift = rng.normal(level, level / 2, size=X.shape[1]).astype(np.float32)
        X = X + shift + rng.normal(0, level / 4, X.shape).astype(np.float32)

    elif mode == "scale":
        scale = rng.uniform(1.0, 1.0 + level, size=X.shape[1]).astype(np.float32)
        X = X * scale

    elif mode == "label_flip":
        flip = rng.random(len(y)) < (level / 2)
        y[flip] = 1 - y[flip]

    else:
        raise ValueError(f"Unknown drift mode: {mode}")

    client.dataloader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long)),
        batch_size=client.batch_size, shuffle=True, drop_last=False,
    )
    client.drifted = True
    return client


if __name__ == "__main__":
    # Sanity check: stationary stream should stay quiet, shifted stream should alarm.
    rng = np.random.default_rng(0)
    ph = PageHinkley(delta=0.005, threshold=0.05, min_rounds=3)

    print("Stationary phase (loss ~0.10):")
    for t in range(10):
        r = ph.update(rng.normal(0.10, 0.005))
        print(f"  t={t+1:2d}  PH={r['ph_stat']:.4f}  alarm={r['detected']}")

    print("Drifted phase (loss ~0.25):")
    for t in range(10, 20):
        r = ph.update(rng.normal(0.25, 0.005))
        print(f"  t={t+1:2d}  PH={r['ph_stat']:.4f}  alarm={r['detected']}")
