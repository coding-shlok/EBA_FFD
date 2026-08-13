"""
data_loader.py
==============
Dataset loading and non-IID partitioning into federated clients.

Changes in v2:
  - Fully parameterised by a config dict (num_clients, alpha, seed, ...) so the
    scaling study (contribution #6) can sweep 4/8/16/32 banks and the
    multi-seed protocol (contribution #5) can vary the seed without editing
    module-level constants.
  - A MINIMUM FRAUD FLOOR per client. The dataset contains only 492 fraud
    records (~394 after the test split). Under a Dirichlet(0.5) draw across 32
    clients, several banks receive zero or one fraud record, which makes SMOTE
    undefined and produces degenerate clients that silently corrupt the scaling
    curve. We redistribute from the fraud-richest clients to guarantee a floor,
    and report how many clients needed topping up.
  - Optional subsampling for smoke tests.

Non-IID Simulation Logic:
  Each client receives a different proportion of fraud/non-fraud samples,
  mimicking real-world heterogeneity across banks. Client class proportions are
  drawn from a Dirichlet distribution; lower alpha = more heterogeneous.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")

# v1 module-level defaults, kept so the original entry points still work
NUM_CLIENTS     = 4
DIRICHLET_ALPHA = 0.5
RANDOM_SEED     = 42
TEST_SIZE       = 0.2


# ── Dataset Download ───────────────────────────────────────────────────────────
def download_dataset():
    """Download creditcard.csv if not already present."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("[data_loader] Downloading Credit Card Fraud dataset...")
        try:
            import urllib.request
            urllib.request.urlretrieve(DATASET_URL, DATA_PATH)
            print(f"[data_loader] Dataset saved to {DATA_PATH}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download dataset: {e}\n"
                f"Please manually download from Kaggle:\n"
                f"  https://www.kaggle.com/mlg-ulb/creditcardfraud\n"
                f"and place it at: {DATA_PATH}"
            )
    else:
        print(f"[data_loader] Dataset found at {DATA_PATH}")


# ── Load & Feature Engineering ─────────────────────────────────────────────────
def load_raw_data(seed=42):
    """
    Load the CSV and engineer features.
      Amount -> log1p (reduces skew)
      Time   -> sin/cos cyclic encoding over a 24h period

    NOTE: no subsampling happens here. Subsampling is applied later, to the
    TRAINING split only (see subsample_training_pool()), so validation and test
    metrics are always computed on the full held-out data regardless of preset.
    """
    df = pd.read_csv(DATA_PATH)

    print(f"[data_loader] Raw data shape: {df.shape}")
    print(f"[data_loader] Fraud ratio: {df['Class'].mean()*100:.4f}%")

    df["Amount_log"] = np.log1p(df["Amount"])

    seconds_in_day = 86400
    df["Time_sin"] = np.sin(2 * np.pi * df["Time"] / seconds_in_day)
    df["Time_cos"] = np.cos(2 * np.pi * df["Time"] / seconds_in_day)

    df.drop(columns=["Time", "Amount"], inplace=True)

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    feature_names = [c for c in df.columns if c != "Class"]
    return X, y, feature_names


# ── Global Train/Test Split ────────────────────────────────────────────────────
def global_train_test_split(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED,
                            val_size=0.1):
    """
    Hold out a global test set BEFORE federated partitioning, plus a VALIDATION
    split carved out of the training data. Both come from the FULL dataset —
    subsampling (see subsample_training_pool()) is applied afterwards, to the
    training split only, so val/test metrics never shrink with the preset.

    Why the validation split exists (added in v2)
    ─────────────────────────────────────────────
    Under real DP-SGD the model's output probabilities are compressed toward the
    prior, so a hard 0.5 decision threshold classifies essentially everything as
    non-fraud even when the ranking is near-perfect (ROC-AUC > 0.99, recall 0.0).
    The decision threshold must therefore be CALIBRATED. It is calibrated here on
    validation data only — never on the test set — so the reported test metrics
    remain honest.
    """
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    # Carve validation out of what remains
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_frac, random_state=seed, stratify=y_tr)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Training-pool subsampling ─────────────────────────────────────────────────
def subsample_training_pool(X_train, y_train, frac, seed):
    """
    Stratified subsample of the TRAINING split only (smoke/quick presets).

    Validation and test are carved out BEFORE this runs (see
    global_train_test_split), so held-out metrics always reflect the full
    dataset's fraud count — 98 test frauds, not ~20 — regardless of how much
    training data a preset uses. Only the federated partitioning and local
    training pool shrink.
    """
    if frac >= 1.0:
        return X_train, y_train

    rng          = np.random.default_rng(seed)
    fraud_idx    = np.where(y_train == 1)[0]
    nonfraud_idx = np.where(y_train == 0)[0]
    n_f  = max(2, int(len(fraud_idx)    * frac))
    n_nf = max(2, int(len(nonfraud_idx) * frac))

    keep = np.concatenate([
        rng.choice(fraud_idx,    n_f,  replace=False),
        rng.choice(nonfraud_idx, n_nf, replace=False),
    ])
    rng.shuffle(keep)

    print(f"[data_loader] Subsampled TRAINING pool to {frac:.1%} "
          f"({len(keep):,} rows, {n_f} fraud)")
    return X_train[keep], y_train[keep]


# ── Minimum fraud floor ────────────────────────────────────────────────────────
def _enforce_fraud_floor(client_fraud_idx, min_fraud, rng):
    """
    Guarantee every client holds at least `min_fraud` fraud records by moving
    records from the fraud-richest clients to the poorest.

    Returns (client_fraud_idx, n_topped_up). If the floor is globally infeasible
    (too many clients for too few fraud records) the floor is lowered and a
    warning is printed rather than failing — the scaling study should still run,
    just with the limitation documented.
    """
    n_clients   = len(client_fraud_idx)
    total_fraud = sum(len(f) for f in client_fraud_idx)

    feasible_floor = total_fraud // n_clients
    if min_fraud > feasible_floor:
        print(f"[data_loader] ! Fraud floor {min_fraud} infeasible for "
              f"{n_clients} clients ({total_fraud} fraud records total). "
              f"Lowering floor to {feasible_floor}.")
        min_fraud = feasible_floor

    client_fraud_idx = [list(f) for f in client_fraud_idx]
    topped_up = 0

    for _ in range(n_clients * 4):    # bounded number of transfer passes
        sizes = [len(f) for f in client_fraud_idx]
        poorest = int(np.argmin(sizes))
        richest = int(np.argmax(sizes))
        if sizes[poorest] >= min_fraud:
            break
        if sizes[richest] <= min_fraud:
            break
        need = min_fraud - sizes[poorest]
        give = min(need, sizes[richest] - min_fraud)
        if give <= 0:
            break
        moved = [client_fraud_idx[richest].pop() for _ in range(give)]
        client_fraud_idx[poorest].extend(moved)
        topped_up += 1

    return [np.array(f, dtype=int) for f in client_fraud_idx], topped_up, min_fraud


# ── Non-IID Federated Partitioning ────────────────────────────────────────────
def partition_non_iid(X_train, y_train, num_clients=NUM_CLIENTS,
                      alpha=DIRICHLET_ALPHA, seed=RANDOM_SEED, min_fraud=0):
    """
    Non-IID partitioning via a Dirichlet distribution drawn independently per
    class, so each client gets a different fraud/non-fraud mix.

    alpha:
      100  -> nearly IID
      0.5  -> moderately non-IID (default)
      0.1  -> extremely heterogeneous
    """
    rng = np.random.default_rng(seed)

    fraud_idx    = np.where(y_train == 1)[0]
    nonfraud_idx = np.where(y_train == 0)[0]
    rng.shuffle(fraud_idx)
    rng.shuffle(nonfraud_idx)

    fraud_props    = rng.dirichlet([alpha] * num_clients)
    nonfraud_props = rng.dirichlet([alpha] * num_clients)

    f_splits  = np.clip((np.cumsum(fraud_props)    * len(fraud_idx)).astype(int),
                        0, len(fraud_idx))
    nf_splits = np.clip((np.cumsum(nonfraud_props) * len(nonfraud_idx)).astype(int),
                        0, len(nonfraud_idx))

    client_fraud, client_nonfraud = [], []
    prev_f = prev_nf = 0
    for i in range(num_clients):
        client_fraud.append(fraud_idx[prev_f:f_splits[i]])
        client_nonfraud.append(nonfraud_idx[prev_nf:nf_splits[i]])
        prev_f, prev_nf = f_splits[i], nf_splits[i]

    topped_up, effective_floor = 0, min_fraud
    if min_fraud > 0:
        client_fraud, topped_up, effective_floor = _enforce_fraud_floor(
            client_fraud, min_fraud, rng)
        if topped_up:
            print(f"[data_loader] Fraud floor applied: {topped_up} client(s) "
                  f"topped up to >= {effective_floor} fraud records")

    client_data = []
    for i in range(num_clients):
        combined = np.concatenate([client_fraud[i], client_nonfraud[i]]).astype(int)
        rng.shuffle(combined)

        # A client with no data at all would break the DataLoader; give it a
        # minimal slice rather than crashing the sweep.
        if len(combined) < 2:
            combined = rng.choice(len(X_train), 2, replace=False)

        X_c, y_c = X_train[combined], y_train[combined]
        print(f"[data_loader] Client {i}: {len(X_c):>7,} samples | "
              f"fraud: {int(y_c.sum()):>4} ({y_c.mean()*100:.4f}%)")

        client_data.append({"X": X_c, "y": y_c, "client_id": i})

    return client_data, {"clients_topped_up": topped_up,
                         "effective_fraud_floor": int(effective_floor)}


# ── Main Entry ─────────────────────────────────────────────────────────────────
def load_federated_data(config=None, skip_download=False):
    """
    Full pipeline: download -> load -> split -> partition.

    config keys used: num_clients, dirichlet_alpha, seed, test_size,
                      subsample_frac, min_fraud_per_client

    Returns a dict with keys:
        clients       : list of {X, y, client_id}
        X_val, y_val  : validation split (threshold calibration only)
        X_test, y_test: global held-out test set
        feature_names : list[str]
        scaler        : fitted StandardScaler
        meta          : partition diagnostics
    """
    config = config or {}
    num_clients = config.get("num_clients",          NUM_CLIENTS)
    alpha       = config.get("dirichlet_alpha",      DIRICHLET_ALPHA)
    seed        = config.get("seed",                 RANDOM_SEED)
    test_size   = config.get("test_size",            TEST_SIZE)
    val_size    = config.get("val_size",             0.1)
    subsample   = config.get("subsample_frac",       1.0)
    min_fraud   = config.get("min_fraud_per_client", 0)

    if not skip_download:
        download_dataset()

    X, y, feature_names = load_raw_data(seed=seed)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        global_train_test_split(X, y, test_size=test_size, seed=seed,
                                val_size=val_size)

    # Subsample the TRAINING pool only — val/test stay at full size (see
    # subsample_training_pool docstring for why this matters at a 0.17% base rate).
    X_train, y_train = subsample_training_pool(X_train, y_train, subsample, seed)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print(f"[data_loader] train: {X_train.shape} | val: {X_val.shape} | "
          f"test: {X_test.shape}")

    client_data, meta = partition_non_iid(
        X_train, y_train, num_clients=num_clients, alpha=alpha,
        seed=seed, min_fraud=min_fraud)

    meta.update({
        "n_train": int(len(X_train)), "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_fraud_train": int(y_train.sum()), "n_fraud_val": int(y_val.sum()),
        "n_fraud_test": int(y_test.sum()),
        "num_clients": num_clients, "dirichlet_alpha": alpha, "seed": seed,
    })

    return {
        "clients": client_data, "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test, "feature_names": feature_names,
        "scaler": scaler, "meta": meta,
    }


if __name__ == "__main__":
    from config import get_config
    cfg  = get_config(num_clients=8, subsample_frac=0.1, min_fraud_per_client=10)
    data = load_federated_data(cfg)
    print(f"\n[data_loader] {len(data['clients'])} clients ready. "
          f"meta={data['meta']}")
