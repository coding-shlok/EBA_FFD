"""
explainability.py
=================
SHAP and LIME explanations for the EBA-FFD fraud detection model.

Why Explainability Matters in Fraud Detection:
  - Regulatory compliance (GDPR Article 22: right to explanation)
  - Fraud analyst trust — they need to understand WHY a transaction is flagged
  - Debugging model failures
  - Feature engineering insights

SHAP (SHapley Additive exPlanations):
  - Based on cooperative game theory (Shapley values)
  - Assigns each feature a contribution to the prediction
  - Global: mean |SHAP| across all predictions → overall feature importance
  - Local: per-transaction explanation → "V28 increased fraud probability by 0.32"
  - We use DeepExplainer (fast NN-specific SHAP approximation)

LIME (Local Interpretable Model-agnostic Explanations):
  - Fits a simple interpretable model (linear) around each prediction
  - Perturbs the input and observes output changes
  - Provides local explanation: which features were most influential
    for THIS specific flagged transaction
  - Model-agnostic: works with any black-box model
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from model import CNNLSTMFraudDetector, build_model, set_model_weights

PLOT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "explanations")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Model Wrapper for SHAP/LIME ────────────────────────────────────────────────
class ModelWrapper:
    """
    Wraps the PyTorch model to expose a predict_proba interface
    compatible with SHAP and LIME (expects numpy in, numpy out).
    """

    def __init__(self, model, device="cpu"):
        self.model  = model
        self.device = device
        self.model.eval()

    def predict_proba(self, X):
        """
        Args:
            X: numpy array (n_samples, n_features)
        Returns:
            proba: (n_samples, 2) — [P(non-fraud), P(fraud)]
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)

        X_t  = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            p_fraud = torch.sigmoid(logits).cpu().numpy()

        p_nonfr = 1.0 - p_fraud
        return np.column_stack([p_nonfr, p_fraud])

    def predict(self, X):
        """Binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


# ── SHAP Analysis ──────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_test, y_test, feature_names, device="cpu",
                      n_background=100, n_explain=50):
    """
    SHAP DeepExplainer for the CNN+LSTM model.

    Args:
        model:         trained CNNLSTMFraudDetector
        X_test:        test features (numpy)
        y_test:        test labels (numpy)
        feature_names: list of feature name strings
        n_background:  number of background samples for SHAP kernel
        n_explain:     number of fraud samples to explain
    """
    try:
        import shap
    except ImportError:
        print("[explainability] SHAP not installed. Run: pip install shap")
        return None

    print("[explainability] Running SHAP analysis...")
    model.eval()

    # Select fraud samples to explain
    fraud_mask  = y_test == 1
    X_fraud     = X_test[fraud_mask][:n_explain]
    X_background = X_test[:n_background]

    # Convert to tensors
    X_bg_t  = torch.tensor(X_background, dtype=torch.float32).to(device)
    X_exp_t = torch.tensor(X_fraud,      dtype=torch.float32).to(device)

    # SHAP DeepExplainer requires a model that takes tensors
    # We wrap the model to return sigmoid probabilities
    class ShapModel(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.sigmoid(self.m(x)).unsqueeze(-1)

    shap_model = ShapModel(model).to(device)
    shap_model.eval()

    try:
        explainer   = shap.DeepExplainer(shap_model, X_bg_t)
        shap_values = explainer.shap_values(X_exp_t)

        # shap_values: list of arrays or single array
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        if sv.ndim == 3:
            sv = sv[:, :, 0]

        print(f"[explainability] SHAP values computed: {sv.shape}")

        # ─ Plot 1: Global Feature Importance (mean |SHAP|) ─────────────────────
        mean_abs_shap = np.abs(sv).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)[::-1][:20]

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_idx)))
        bars = ax.barh(
            range(len(sorted_idx)),
            mean_abs_shap[sorted_idx],
            color=colors, edgecolor="white", linewidth=0.5
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx],
                           color="white", fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|", color="white", fontsize=12)
        ax.set_title("Global Feature Importance (SHAP) — Top 20 Features",
                     color="white", fontsize=13)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "shap_global_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[explainability] SHAP global importance → {path}")

        # ─ Plot 2: SHAP Beeswarm / Summary ────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(11, 8))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#1a1d27")

        # Beeswarm-style scatter of SHAP values
        top_features = sorted_idx[:15]
        for yi, feat_idx in enumerate(reversed(top_features)):
            shap_col = sv[:, feat_idx]
            feat_col = X_fraud[:len(sv), feat_idx]
            # Normalize feature values for color
            norm_feat = (feat_col - feat_col.min()) / (feat_col.ptp() + 1e-8)
            scatter = ax2.scatter(
                shap_col,
                np.full(len(shap_col), yi) + np.random.normal(0, 0.1, len(shap_col)),
                c=norm_feat, cmap="RdBu_r", alpha=0.6, s=12
            )

        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(
            [feature_names[i] for i in reversed(top_features)],
            color="white", fontsize=10
        )
        ax2.axvline(0, color="white", linestyle="--", alpha=0.4)
        ax2.set_xlabel("SHAP Value (impact on fraud probability)", color="white", fontsize=11)
        ax2.set_title("SHAP Beeswarm — Fraud Sample Explanations",
                      color="white", fontsize=13)
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#444")

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Feature Value (normalized)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        plt.tight_layout()
        path2 = os.path.join(PLOT_DIR, "shap_beeswarm.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"[explainability] SHAP beeswarm → {path2}")

        # ─ Plot 3: Local explanation for top fraud sample ──────────────────────
        _plot_shap_local(sv[0], X_fraud[0], feature_names, sample_id=0)

        return sv, sorted_idx

    except Exception as e:
        print(f"[explainability] SHAP DeepExplainer failed: {e}")
        print("[explainability] Falling back to KernelExplainer...")
        return _run_shap_kernel(model, X_test, y_test, feature_names, device,
                                n_background, n_explain)


def _run_shap_kernel(model, X_test, y_test, feature_names, device,
                     n_background=50, n_explain=20):
    """Fallback: SHAP KernelExplainer (model-agnostic but slower)."""
    import shap
    wrapper = ModelWrapper(model, device)

    X_background = shap.sample(X_test, n_background)
    X_fraud      = X_test[y_test == 1][:n_explain]

    explainer   = shap.KernelExplainer(
        lambda x: wrapper.predict_proba(x)[:, 1],
        X_background
    )
    shap_values = explainer.shap_values(X_fraud, nsamples=50)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1][:15]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx],
            color="#10B981", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], color="white")
    ax.set_xlabel("Mean |SHAP Value|", color="white", fontsize=12)
    ax.set_title("SHAP Global Feature Importance (Kernel)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "shap_global_kernel.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[explainability] SHAP kernel importance → {path}")
    return shap_values, sorted_idx


def _plot_shap_local(shap_vals, x_sample, feature_names, sample_id=0):
    """Force-plot style local explanation for a single fraud transaction."""
    top_n  = 15
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    vals   = shap_vals[sorted_idx]
    names  = [f"{feature_names[i]}\n={x_sample[i]:.3f}" for i in sorted_idx]
    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in vals]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (→ increases fraud probability)", color="white", fontsize=11)
    ax.set_title(f"Local SHAP Explanation — Fraud Sample #{sample_id}",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#EF4444", label="Increases fraud prob"),
              Patch(facecolor="#3B82F6", label="Decreases fraud prob")]
    ax.legend(handles=legend, facecolor="#2a2d3a", labelcolor="white", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"shap_local_sample{sample_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[explainability] SHAP local explanation → {path}")


# ── LIME Analysis ──────────────────────────────────────────────────────────────
def run_lime_analysis(model, X_test, y_test, feature_names, device="cpu",
                      n_samples_lime=500, n_fraud_to_explain=5):
    """
    LIME explanations for flagged fraud transactions.

    LIME fits a local linear model around each prediction by:
      1. Perturbing the input features
      2. Getting model predictions on perturbations
      3. Weighting perturbations by proximity to original
      4. Fitting a weighted linear model
      5. Linear model coefficients = feature importances

    Args:
        n_samples_lime: number of perturbation samples for each explanation
        n_fraud_to_explain: number of fraud transactions to explain
    """
    try:
        from lime import lime_tabular
    except ImportError:
        print("[explainability] LIME not installed. Run: pip install lime")
        return

    print(f"\n[explainability] Running LIME analysis on {n_fraud_to_explain} fraud samples...")

    wrapper = ModelWrapper(model, device)

    # LIME tabular explainer — trained on background distribution
    explainer = lime_tabular.LimeTabularExplainer(
        training_data    = X_test[:500],   # Background reference distribution
        feature_names    = feature_names,
        class_names      = ["Non-Fraud", "Fraud"],
        mode             = "classification",
        discretize_continuous = True,
        random_state     = 42,
    )

    # Select fraud samples with high model confidence
    fraud_idx = np.where(y_test == 1)[0]
    probs     = wrapper.predict_proba(X_test[fraud_idx])[:, 1]
    top_fraud = fraud_idx[np.argsort(probs)[::-1][:n_fraud_to_explain]]

    for sample_num, idx in enumerate(top_fraud):
        x_sample = X_test[idx]
        prob     = probs[np.where(fraud_idx == idx)[0][0]]

        print(f"  Explaining fraud sample {idx} (P(fraud)={prob:.4f})")

        explanation = explainer.explain_instance(
            data_row       = x_sample,
            predict_fn     = wrapper.predict_proba,
            num_features   = 15,
            num_samples    = n_samples_lime,
            labels         = (1,)
        )

        # Extract feature importances for fraud class
        exp_list = explanation.as_list(label=1)
        feat_names_lime = [e[0] for e in exp_list]
        feat_vals_lime  = [e[1] for e in exp_list]

        colors = ["#EF4444" if v > 0 else "#3B82F6" for v in feat_vals_lime]

        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")

        y_pos = range(len(feat_names_lime))
        ax.barh(y_pos, feat_vals_lime, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names_lime, color="white", fontsize=9)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Feature Importance (LIME)", color="white", fontsize=11)
        ax.set_title(
            f"LIME Local Explanation — Fraud Transaction #{idx}\n"
            f"Model P(fraud) = {prob:.4f}",
            color="white", fontsize=12
        )
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor="#EF4444", label="Supports fraud prediction"),
            Patch(facecolor="#3B82F6", label="Contradicts fraud prediction")
        ]
        ax.legend(handles=legend_elems, facecolor="#2a2d3a",
                  labelcolor="white", fontsize=10)

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"lime_explanation_sample{sample_num}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[explainability]   → {path}")

    print("[explainability] LIME analysis complete.")


# ── Feature Correlation Heatmap ────────────────────────────────────────────────
def plot_fraud_feature_comparison(X_test, y_test, feature_names, top_n=15):
    """
    Compare mean feature values for fraud vs non-fraud transactions.
    Quick visual inspection of which features differ most between classes.
    """
    X_fraud   = X_test[y_test == 1]
    X_normal  = X_test[y_test == 0]

    mean_fraud  = np.abs(X_fraud.mean(axis=0))
    mean_normal = np.abs(X_normal.mean(axis=0))

    # Top N features by fraud-normal difference
    diff = np.abs(mean_fraud - mean_normal)
    top_idx = np.argsort(diff)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    x    = np.arange(top_n)
    w    = 0.35
    ax.bar(x - w/2, mean_fraud[top_idx],  w, label="Fraud",     color="#EF4444", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, mean_normal[top_idx], w, label="Non-Fraud", color="#3B82F6", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in top_idx],
                       rotation=45, ha="right", color="white", fontsize=9)
    ax.set_ylabel("|Mean Feature Value|", color="white", fontsize=11)
    ax.set_title(f"Top {top_n} Features: Fraud vs Non-Fraud Mean Values",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=11)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fraud_feature_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[explainability] Feature comparison → {path}")


# ── Main entry ─────────────────────────────────────────────────────────────────
def run_explainability(model_weights_path, X_test, y_test, feature_names,
                       input_dim=30, device="cpu", config=None):
    """
    Full explainability pipeline: SHAP + LIME + feature comparison.

    Args:
        model_weights_path: path to saved model state dict
        X_test, y_test:     test data
        feature_names:      list of feature names
    """
    print("\n[explainability] === Explainability Module ===")

    # Load model. The architecture MUST match the one that produced the
    # checkpoint: under DP the model uses DPLSTM + GroupNorm, whose state_dict
    # keys differ from nn.LSTM + BatchNorm. Passing the same config used for
    # training guarantees the keys line up.
    model = build_model(input_dim=input_dim, device=device, config=config)
    weights = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    print(f"[explainability] Loaded model from {model_weights_path}")

    # Feature comparison plot (fast, no packages needed)
    plot_fraud_feature_comparison(X_test, y_test, feature_names)

    # SHAP
    run_shap_analysis(model, X_test, y_test, feature_names, device=device)

    # LIME
    run_lime_analysis(model, X_test, y_test, feature_names, device=device)

    print("\n[explainability] All explanations saved to outputs/explanations/")


if __name__ == "__main__":
    import os
    model_path = os.path.join(
        os.path.dirname(__file__), "outputs", "models", "best_global_model.pt"
    )
    if not os.path.exists(model_path):
        print("No trained model found. Run training_loop.py first.")
    else:
        from data_loader import load_federated_data
        from config import get_config
        cfg  = get_config()
        data = load_federated_data(cfg)
        run_explainability(model_path, data["X_test"], data["y_test"],
                           data["feature_names"],
                           input_dim=data["X_test"].shape[1],
                           device=cfg["device"], config=cfg)
