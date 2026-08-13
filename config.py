"""
config.py
=========
Single source of truth for every experimental setting in EBA-FFD v2.

All experiment drivers (training_loop, baselines, ablations, scaling,
run_all_experiments) build their configuration by copying BASE_CONFIG and
overriding only the keys they care about. This guarantees that an ablation
differs from the main run in exactly one dimension, which is what makes the
ablation table interpretable in a paper.
"""

import copy
import torch


BASE_CONFIG = {
    # ── Federated training ────────────────────────────────────────────────────
    "num_rounds":            10,
    "local_epochs":          3,
    "batch_size":            256,
    "lr":                    1e-3,
    "num_clients":           4,
    "dirichlet_alpha":       0.5,

    # ── Aggregation strategy ──────────────────────────────────────────────────
    # "fedavg"   : classic sample-weighted FedAvg (McMahan et al., 2017)
    # "adaptive" : drift-aware FedAvg  (contribution #1)
    # "fedprox"  : FedAvg aggregation + proximal client objective (Li et al., 2020)
    "aggregation":           "adaptive",
    "fedprox_mu":            0.01,

    # ── Adaptive aggregation / drift detection (contribution #1) ──────────────
    "drift_detector": {
        "delta":             0.005,   # PH allowed magnitude of change
        "threshold":         0.05,    # PH alarm threshold (lambda)
        "alpha":             0.9999,  # forgetting factor
        "min_rounds":        3,       # warm-up before alarms may fire
    },
    "trust_beta":            2.0,     # steepness of trust decay vs drift severity
    "trust_min":             0.20,    # a drifting bank is damped, never silenced
    "trust_recovery":        0.30,    # per-round recovery toward full trust

    # ── Differential privacy (contribution #2) ────────────────────────────────
    # "opacus" : true DP-SGD, per-sample clipping + RDP accountant
    # "none"   : no privacy mechanism (ablation / upper bound)
    "dp_mode":               "opacus",
    "target_epsilon":        3.0,     # per-client budget over ALL rounds
    "max_grad_norm":         1.0,     # per-sample clipping bound C
    "delta_rule":            "1/n",   # delta_k = 1 / n_k  (capped at 1e-5)

    # ── Model architecture (ablation switches) ────────────────────────────────
    # use_lstm defaults to False: a capacity sweep under fixed DP-SGD (eps=3.0,
    # see capacity_sweep.py) showed every architecture keeping the BiLSTM block
    # scored PR-AUC 0.33-0.46, while every architecture dropping it scored
    # 0.64-0.70, regardless of parameter count. DP-SGD noise on the DPLSTM's
    # per-timestep per-sample gradients dominates its own signal on this
    # dataset, which has no genuine per-account sequence for the LSTM to model
    # in the first place (see README "Known limitations" #5). The ablation
    # variant "with_lstm" (ablations.py) reproduces the discarded architecture
    # for the paper's record of why it was cut.
    "model": {
        "use_cnn":           True,
        "use_lstm":          False,
        "cnn_channels":      64,
        "cnn_kernel":        3,
        "lstm_hidden":       64,
        "lstm_layers":       2,
        "dropout":           0.3,
        # GroupNorm is mandatory under DP-SGD: BatchNorm leaks across samples
        # and is rejected by Opacus' ModuleValidator.
        "norm_type":         "group",
        "dp_lstm":           True,    # opacus.layers.DPLSTM instead of nn.LSTM
    },

    # ── Loss (ablation switch) ────────────────────────────────────────────────
    "loss":                  "focal",   # "focal" | "bce"
    "focal_alpha":           0.25,
    "focal_gamma":           2.0,

    # ── Class balancing (ablation switch) ─────────────────────────────────────
    # "kmeans_smote_enn" | "smote_enn" | "smote" | "none"
    "balancing":             "kmeans_smote_enn",
    "sampling_strategy":     0.3,

    # ── Data ──────────────────────────────────────────────────────────────────
    "test_size":             0.2,
    "subsample_frac":        1.0,   # <1.0 subsamples the dataset (smoke tests)
    "min_fraud_per_client":  10,    # floor enforced when scaling to 16/32 banks

    # ── Statistical protocol (contribution #5) ────────────────────────────────
    "seeds":                 [42, 43, 44],
    "seed":                  42,    # seed of a single run

    # ── Concept-drift injection (used to VALIDATE contribution #1) ────────────
    "drift_enabled":         False,
    "drift_round":           5,          # round at which drift begins
    "drift_clients":         [0],        # which banks drift
    "drift_mode":            "covariate_shift",  # covariate_shift|label_flip|scale
    "drift_level":           0.5,

    # ── Misc ──────────────────────────────────────────────────────────────────
    "device":                "cuda" if torch.cuda.is_available() else "cpu",
    "visualize":             True,
    "verbose":               True,
}


def get_config(**overrides):
    """
    Return a deep copy of BASE_CONFIG with top-level keys overridden.

    Nested dicts ("model", "drift_detector") are merged rather than replaced,
    so get_config(model={"use_lstm": False}) keeps every other model setting.
    """
    cfg = copy.deepcopy(BASE_CONFIG)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value

    # ── Consistency guard ─────────────────────────────────────────────────────
    # Opacus cannot differentiate through nn.LSTM or BatchNorm. If DP is off we
    # are free to use the faster standard layers; if DP is on we must not.
    if cfg["dp_mode"] == "opacus":
        cfg["model"]["dp_lstm"]   = True
        cfg["model"]["norm_type"] = "group"

    return cfg


# Backwards compatibility with the original v1 entry points
CONFIG = get_config()
