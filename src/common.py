"""Shared data interface for every model in the CTR lab."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
RESULTS.mkdir(exist_ok=True)
FEATURES = [f"feat_{i}" for i in range(1, 23)]  # 22 anonymised Avazu fields (ids already integer-encoded upstream)

def load(split, limit=None):
    """split in {'train','valid','test'}; returns DataFrame with 'label' + FEATURES."""
    df = pd.read_parquet(DATA / f"{split}.parquet")
    return df.head(limit) if limit else df

def build_vocab(train_df, min_count=2):
    """Per-field map raw id -> contiguous index (0 = rare/unseen bucket). Fit on TRAIN only (no leakage)."""
    vocab = {}
    for c in FEATURES:
        vc = train_df[c].value_counts()
        keep = vc[vc >= min_count].index
        vocab[c] = pd.Series(np.arange(1, len(keep) + 1), index=keep)
    return vocab

def encode(df, vocab):
    """Returns int64 array [n, 22] of per-field contiguous indices (0 for unseen)."""
    return np.stack([df[c].map(vocab[c]).fillna(0).astype("int64").to_numpy() for c in FEATURES], axis=1)

def train_ctr():
    """Click rate of the full train split: the honest 'no model' prior (known before seeing test)."""
    return float(pd.read_parquet(DATA / "train.parquet", columns=["label"])["label"].mean())

def metrics(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    base = np.full_like(p, train_ctr())
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p)),
            "logloss_constant": float(log_loss(y, base)), "n": int(len(y))}

def save_result(name, y_test, p_test, extra):
    """Every model calls this once: writes results/<name>.json and results/preds_<name>.npy (test-set probabilities)."""
    np.save(RESULTS / f"preds_{name}.npy", np.asarray(p_test, dtype="float32"))
    out = {"model": name, **metrics(np.asarray(y_test), np.asarray(p_test)), **extra}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out
