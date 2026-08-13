"""
privacy.py
==========
Differential privacy for EBA-FFD  (contribution #2).

What changed from v1
────────────────────
v1 clipped *gradients* during training and then added Gaussian noise to the
*weights* afterwards, with std = noise_multiplier * max_grad_norm. Those two
operations were unconnected: clipping the gradient of a minibatch does not
bound the sensitivity of the resulting weight vector to a single training
record, so no (epsilon, delta) statement follows from that noise. The v1
mechanism was a heuristic regulariser, not differential privacy.

v2 implements genuine DP-SGD (Abadi et al., 2016) via Opacus:

  1. PER-SAMPLE gradient clipping. Each individual record's gradient is clipped
     to L2 norm <= C *before* averaging. This is what actually bounds
     sensitivity to one record.
  2. Gaussian noise N(0, sigma^2 C^2) added to the SUM of clipped per-sample
     gradients, once per optimiser step.
  3. Poisson subsampling, so the privacy amplification-by-subsampling argument
     the accountant relies on is actually valid.
  4. An RDP accountant that composes the privacy loss across every step of
     every round and reports the realised epsilon.

Per-bank noise calibration
──────────────────────────
Banks hold different amounts of data. Under a fixed noise multiplier they would
end up with wildly different privacy guarantees: a small bank takes more privacy
loss per record than a large one, because each record participates in a larger
fraction of the batches.

We invert the problem. Every bank is given the SAME privacy guarantee
epsilon_target, and we solve for the noise multiplier sigma_k that achieves it
given that bank's own sampling rate q_k = B / n_k and its own step count
T_k = rounds * epochs * ceil(n_k / B):

    sigma_k = argmin { sigma : RDP_accountant(sigma, q_k, T_k) <= epsilon_target }

delta_k is set per bank as min(1/n_k, 1e-5), following the standard convention
that delta should be well below 1/n_k. Small banks therefore receive MORE noise
per step than large banks — which is the correct direction, and is what
"custom-tuned noise level per bank" means here.

Architectural constraints DP-SGD imposes
────────────────────────────────────────
Per-sample gradients must be well-defined, which rules out two layers used in v1:

  BatchNorm1d -> replaced with GroupNorm. BatchNorm's statistics mix information
                 across samples in a batch, so one record's gradient is not
                 separable, and Opacus' ModuleValidator rejects it outright.
  nn.LSTM     -> replaced with opacus.layers.DPLSTM, a functionally equivalent
                 re-implementation that exposes per-timestep per-sample grads.

Both swaps are handled in model.py and switched on automatically by config.py
whenever dp_mode == "opacus".
"""

import math
import warnings

import torch

try:
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:                                        # pragma: no cover
    OPACUS_AVAILABLE = False
    warnings.warn("Opacus not installed — DP mode 'opacus' will be unavailable.")


# ── Budget bookkeeping ────────────────────────────────────────────────────────
def compute_delta(n_samples):
    """
    delta should sit comfortably below 1/n. We use min(1/n, 1e-5), which is the
    convention in the DP-SGD literature and keeps delta meaningful for the
    smallest banks in the 32-client setting.
    """
    if n_samples <= 0:
        return 1e-5
    return min(1.0 / n_samples, 1e-5)


def compute_steps(n_samples, batch_size, local_epochs, num_rounds):
    """
    Total number of noisy optimiser steps a client will take across the ENTIRE
    federation run. The budget must be calibrated against this total, not
    against a single round, or the composed epsilon will overshoot by ~T times.
    """
    steps_per_epoch = max(1, math.ceil(n_samples / batch_size))
    return steps_per_epoch * local_epochs * num_rounds


def calibrate_noise_multiplier(target_epsilon, n_samples, batch_size,
                               local_epochs, num_rounds, delta=None,
                               accountant="rdp"):
    """
    Solve for the smallest sigma achieving target_epsilon for THIS client.

    Returns (sigma, delta, sample_rate, steps).
    """
    if not OPACUS_AVAILABLE:
        raise RuntimeError("Opacus is required for DP calibration.")

    delta       = delta if delta is not None else compute_delta(n_samples)
    sample_rate = min(1.0, batch_size / max(1, n_samples))
    steps       = compute_steps(n_samples, batch_size, local_epochs, num_rounds)

    sigma = get_noise_multiplier(
        target_epsilon = target_epsilon,
        target_delta   = delta,
        sample_rate    = sample_rate,
        steps          = steps,
        accountant     = accountant,
    )
    return float(sigma), float(delta), float(sample_rate), int(steps)


def build_privacy_plan(client_sizes, config):
    """
    Produce the full per-bank privacy plan up front, before training starts.

    Returned records feed directly into the paper's privacy table:
        client | n_k | q_k | steps | sigma_k | epsilon | delta_k
    """
    plan = []
    for cid, n in enumerate(client_sizes):
        if config["dp_mode"] != "opacus":
            plan.append({
                "client_id": cid, "n_samples": int(n), "sigma": 0.0,
                "epsilon": float("inf"), "delta": 0.0,
                "sample_rate": 0.0, "steps": 0, "dp": False,
            })
            continue

        sigma, delta, q, steps = calibrate_noise_multiplier(
            target_epsilon = config["target_epsilon"],
            n_samples      = n,
            batch_size     = config["batch_size"],
            local_epochs   = config["local_epochs"],
            num_rounds     = config["num_rounds"],
        )
        plan.append({
            "client_id":   cid,
            "n_samples":   int(n),
            "sigma":       sigma,
            "epsilon":     float(config["target_epsilon"]),
            "delta":       delta,
            "sample_rate": q,
            "steps":       steps,
            "dp":          True,
        })
    return plan


def print_privacy_plan(plan, target_epsilon):
    """Human-readable summary of the calibration, printed once per run."""
    if not plan or not plan[0].get("dp", False):
        print("[privacy] DP DISABLED — no formal guarantee (ablation / upper bound).")
        return

    print(f"\n[privacy] DP-SGD plan — every bank calibrated to epsilon = {target_epsilon}")
    print(f"[privacy] {'bank':>5} {'n_k':>9} {'q_k':>9} {'steps':>8} "
          f"{'sigma_k':>9} {'delta_k':>10}")
    for r in plan:
        print(f"[privacy] {r['client_id']:>5} {r['n_samples']:>9,} "
              f"{r['sample_rate']:>9.5f} {r['steps']:>8} "
              f"{r['sigma']:>9.4f} {r['delta']:>10.2e}")
    sigmas = [r["sigma"] for r in plan]
    print(f"[privacy] sigma ranges {min(sigmas):.4f} – {max(sigmas):.4f} "
          f"(smaller banks receive more noise, as expected)\n")


# ── Model preparation ─────────────────────────────────────────────────────────
def validate_and_fix_model(model):
    """
    Ensure the module graph is DP-compatible. Raises early with a clear message
    rather than failing deep inside a training loop.
    """
    if not OPACUS_AVAILABLE:
        return model

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"[privacy] ModuleValidator flagged {len(errors)} incompatible "
              f"layer(s); applying automatic fixes.")
        model = ModuleValidator.fix(model)
        remaining = ModuleValidator.validate(model, strict=False)
        if remaining:
            raise RuntimeError(f"Model is not DP-compatible: {remaining}")
    return model


def make_private(model, optimizer, data_loader, sigma, max_grad_norm,
                 accountant=None):
    """
    Attach DP-SGD machinery for one round of local training.

    A fresh PrivacyEngine is created each round, but the RDP accountant object
    is CARRIED OVER between rounds so privacy loss composes across the whole
    federation run rather than resetting every round. Passing accountant=None
    starts a new budget.

    Returns (dp_model, dp_optimizer, dp_loader, engine).
    """
    if not OPACUS_AVAILABLE:
        raise RuntimeError("Opacus is required for DP training.")

    engine = PrivacyEngine(accountant="rdp")
    if accountant is not None:
        engine.accountant = accountant     # resume the existing budget

    dp_model, dp_optimizer, dp_loader = engine.make_private(
        module           = model,
        optimizer        = optimizer,
        data_loader      = data_loader,
        noise_multiplier = sigma,
        max_grad_norm    = max_grad_norm,
        poisson_sampling = True,           # required for the amplification bound
    )
    return dp_model, dp_optimizer, dp_loader, engine


def unwrap(model):
    """
    Recover the plain nn.Module from Opacus' GradSampleModule wrapper so its
    state_dict keys match the server's global model.
    """
    inner = getattr(model, "_module", model)
    return inner


def spent_epsilon(engine, delta):
    """Realised epsilon consumed so far, per the RDP accountant."""
    try:
        return float(engine.get_epsilon(delta))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    print("[privacy] Opacus available:", OPACUS_AVAILABLE)

    # Demonstrate that smaller banks get more noise for the same epsilon.
    cfg = {"dp_mode": "opacus", "target_epsilon": 3.0, "batch_size": 256,
           "local_epochs": 3, "num_rounds": 10}
    plan = build_privacy_plan([50_000, 20_000, 5_000, 1_200], cfg)
    print_privacy_plan(plan, cfg["target_epsilon"])
