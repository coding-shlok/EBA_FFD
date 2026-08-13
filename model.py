"""
model.py
========
CNN + BiLSTM hybrid fraud detector, with architecture switches for ablation
studies (contribution #4) and DP-compatible layer variants (contribution #2).

Architecture Rationale:
  CNN Block:
    - 1D convolutions over the transaction feature vector
    - Extracts local feature patterns (e.g. correlated PCA components)
    - Reduces sequence length via max-pooling

  BiLSTM Block:
    - Models dependencies across the convolved feature positions
    - In production: processes transaction sequences per account
    - In this prototype: processes CNN output as a "sequence" of feature maps

  Dense Head:
    - Binary classification, single logit, dropout regularisation

Ablation switches (config["model"]):
  use_cnn=False   -> features are fed to the LSTM directly as a length-D
                     sequence of scalars. Isolates the CNN's contribution.
  use_lstm=False  -> the CNN feature map is flattened straight into the dense
                     head. Isolates the recurrent block's contribution.
  both False      -> plain MLP on the raw feature vector (lower bound).

DP-compatibility switches:
  norm_type="group" -> GroupNorm instead of BatchNorm1d. BatchNorm's batch
                       statistics couple samples together, so per-sample
                       gradients are not well-defined and Opacus rejects it.
  dp_lstm=True      -> opacus.layers.DPLSTM instead of nn.LSTM. Functionally
                       equivalent, but exposes per-sample gradients. Slower,
                       so it is only enabled when DP is actually on.

Loss — Focal Loss:
  Standard BCE is dominated by the majority class. Focal Loss down-weights easy
  negatives and concentrates gradient on hard positives. gamma controls the
  down-weighting of easy examples; alpha handles class frequency imbalance.
  The "bce" option exists purely as an ablation.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from opacus.layers import DPLSTM
    DPLSTM_AVAILABLE = True
except ImportError:                                        # pragma: no cover
    DPLSTM_AVAILABLE = False


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: weight on the positive (fraud) class
    gamma: focusing parameter; gamma=0 reduces to weighted BCE
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none")
        p_t          = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t      = self.alpha * targets.float() + \
                       (1 - self.alpha) * (1 - targets.float())
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCELoss(nn.Module):
    """Plain BCE-with-logits. Ablation control for Focal Loss."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pw)


def build_loss(config):
    """Instantiate the loss named in config['loss']."""
    if config.get("loss", "focal") == "bce":
        return BCELoss()
    return FocalLoss(alpha=config.get("focal_alpha", 0.25),
                     gamma=config.get("focal_gamma", 2.0))


# ── Normalisation helper ───────────────────────────────────────────────────────
def _make_norm(norm_type, num_channels):
    """
    GroupNorm is the DP-safe choice. We pick a group count that divides the
    channel count, falling back to a single group if none do.
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "none":
        return nn.Identity()

    for g in (8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# ── CNN + BiLSTM Hybrid ────────────────────────────────────────────────────────
class CNNLSTMFraudDetector(nn.Module):
    """
    CNN -> BiLSTM -> Dense hybrid, with both blocks independently removable.

    Input:  (batch, input_dim)
    Output: (batch,) single logit
    """

    def __init__(
        self,
        input_dim:    int   = 30,
        cnn_channels: int   = 64,
        cnn_kernel:   int   = 3,
        lstm_hidden:  int   = 64,
        lstm_layers:  int   = 2,
        dropout:      float = 0.3,
        use_cnn:      bool  = True,
        use_lstm:     bool  = True,
        norm_type:    str   = "group",
        dp_lstm:      bool  = False,
    ):
        super().__init__()

        self.input_dim    = input_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden  = lstm_hidden
        self.use_cnn      = use_cnn
        self.use_lstm     = use_lstm
        self.dp_lstm      = dp_lstm

        # ── CNN Block ─────────────────────────────────────────────────────────
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(1, cnn_channels, kernel_size=cnn_kernel,
                          padding=cnn_kernel // 2),
                _make_norm(norm_type, cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=cnn_kernel,
                          padding=cnn_kernel // 2),
                _make_norm(norm_type, cnn_channels * 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )
            seq_len  = input_dim // 2
            seq_feat = cnn_channels * 2
        else:
            self.cnn = None
            # Raw features become a length-D sequence of scalars
            seq_len  = input_dim
            seq_feat = 1

        # ── Recurrent Block ───────────────────────────────────────────────────
        if use_lstm:
            if dp_lstm:
                if not DPLSTM_AVAILABLE:
                    raise RuntimeError(
                        "dp_lstm=True but opacus.layers.DPLSTM is unavailable.")
                lstm_cls = DPLSTM
            else:
                lstm_cls = nn.LSTM

            self.lstm = lstm_cls(
                input_size    = seq_feat,
                hidden_size   = lstm_hidden,
                num_layers    = lstm_layers,
                batch_first   = True,
                dropout       = dropout if lstm_layers > 1 else 0.0,
                bidirectional = True,
            )
            head_in = lstm_hidden * 2
        else:
            self.lstm = None
            head_in   = seq_len * seq_feat     # flattened feature map

        self.head_in = head_in

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for stable training."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _features(self, x):
        """Shared trunk: returns the representation fed to the dense head."""
        if self.use_cnn:
            x = x.unsqueeze(1)              # (B, 1, D)
            x = self.cnn(x)                 # (B, 2C, D//2)
            x = x.permute(0, 2, 1)          # (B, D//2, 2C)
        else:
            x = x.unsqueeze(-1)             # (B, D, 1)

        if self.use_lstm:
            x, _ = self.lstm(x)             # (B, L, 2H)
            x = x[:, -1, :]                 # last timestep
        else:
            x = x.reshape(x.size(0), -1)    # flatten

        return x

    def forward(self, x):
        return self.classifier(self._features(x)).squeeze(-1)

    def get_embedding(self, x):
        """Penultimate representation — used by SHAP/LIME analysis."""
        return self._features(x)


# ── Model Factory ──────────────────────────────────────────────────────────────
def build_model(input_dim=30, device="cpu", config=None):
    """
    Create a model. If a full config dict is passed, its "model" sub-dict drives
    the architecture; otherwise the v1 defaults are used.
    """
    mcfg = (config or {}).get("model", {}) if config else {}
    model = CNNLSTMFraudDetector(
        input_dim    = input_dim,
        cnn_channels = mcfg.get("cnn_channels", 64),
        cnn_kernel   = mcfg.get("cnn_kernel", 3),
        lstm_hidden  = mcfg.get("lstm_hidden", 64),
        lstm_layers  = mcfg.get("lstm_layers", 2),
        dropout      = mcfg.get("dropout", 0.3),
        use_cnn      = mcfg.get("use_cnn", True),
        use_lstm     = mcfg.get("use_lstm", True),
        norm_type    = mcfg.get("norm_type", "group"),
        dp_lstm      = mcfg.get("dp_lstm", False),
    )
    return model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_weights(model):
    """State dict as CPU tensors, for federated aggregation."""
    inner = getattr(model, "_module", model)   # unwrap Opacus GradSampleModule
    return copy.deepcopy({k: v.cpu() for k, v in inner.state_dict().items()})


def set_model_weights(model, weights):
    inner = getattr(model, "_module", model)
    inner.load_state_dict(weights)
    return model


# ── Quick Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import get_config

    device = "cpu"
    dummy  = torch.randn(32, 31)
    labels = torch.randint(0, 2, (32,))

    variants = {
        "full (CNN+BiLSTM)": {"use_cnn": True,  "use_lstm": True,  "dp_lstm": True},
        "no CNN":            {"use_cnn": False, "use_lstm": True,  "dp_lstm": True},
        "no LSTM":           {"use_cnn": True,  "use_lstm": False, "dp_lstm": True},
        "MLP only":          {"use_cnn": False, "use_lstm": False, "dp_lstm": True},
    }

    for name, override in variants.items():
        cfg  = get_config(model=override)
        m    = build_model(input_dim=31, device=device, config=cfg)
        out  = m(dummy)
        loss = build_loss(cfg)(out, labels)
        print(f"{name:22s} params={count_parameters(m):>9,}  "
              f"out={tuple(out.shape)}  loss={loss.item():.4f}")
