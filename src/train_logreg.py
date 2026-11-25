"""Model "logreg": L2 logistic regression on sparse one-hot of encode() indices (the 2013 ad-click baseline).

C is picked from a small grid on VALID logloss only. TEST is scored once, at the end.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, build_vocab, encode, save_result, FEATURES  # noqa: E402

SEED = 42
C_GRID = [0.001, 0.003, 0.01]


def to_csr(X, offsets, n_cols):
    """X: int64 [n, 22] per-field indices; offsets: per-field start column. Returns CSR with 22 ones per row."""
    n, f = X.shape
    assert n_cols < 2**31, "column ids would overflow int32"
    cols = (X + offsets[None, :]).astype(np.int32).ravel()
    indptr = np.arange(0, n * f + 1, f, dtype=np.int64)
    data = np.ones(n * f, dtype=np.float32)
    return sp.csr_matrix((data, cols, indptr), shape=(n, n_cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="rows per split (smoke test)")
    args = ap.parse_args()
    np.random.seed(SEED)

    # Load and encode one split at a time to keep peak RAM low.
    tr = load("train", args.limit)
    vocab = build_vocab(tr)  # fit on TRAIN only
    sizes = np.array([len(vocab[c]) + 1 for c in FEATURES], dtype=np.int64)  # +1 for the unseen bucket 0
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    n_cols = int(sizes.sum())
    ytr = tr["label"].to_numpy()
    Xtr = to_csr(encode(tr, vocab), offsets, n_cols)
    del tr
    va = load("valid", args.limit)
    yva = va["label"].to_numpy()
    Xva = to_csr(encode(va, vocab), offsets, n_cols)
    del va
    te = load("test", args.limit)
    yte = te["label"].to_numpy()
    Xte = to_csr(encode(te, vocab), offsets, n_cols)
    del te
    print(f"features: {n_cols} one-hot columns, train rows {Xtr.shape[0]}", flush=True)

    best = None  # (valid_logloss, C, model, train_seconds)
    total_secs, per_c = 0.0, {}
    for C in C_GRID:
        t0 = time.time()
        m = LogisticRegression(C=C, solver="liblinear", max_iter=200, random_state=SEED,
                               intercept_scaling=10.0)  # large scaling: bias is barely penalized
        m.fit(Xtr, ytr)
        secs = time.time() - t0
        total_secs += secs
        n_it = int(np.max(m.n_iter_))
        per_c[str(C)] = {"seconds": float(secs), "n_iter": n_it, "hit_max_iter": bool(n_it >= m.max_iter)}
        ll = log_loss(yva, np.clip(m.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7))
        print(f"C={C}: valid logloss {ll:.5f} ({secs:.1f}s, n_iter {n_it})", flush=True)
        if best is None or ll < best[0]:
            best = (ll, C, m, secs)

    _, C, model, secs = best
    p_test = model.predict_proba(Xte)[:, 1]  # TEST touched once
    extra = {"C": C, "train_seconds": float(secs), "train_seconds_total_grid": float(total_secs),
             "per_C": per_c, "intercept_scaling": 10.0, "n_params": int(model.coef_.size + model.intercept_.size)}
    save_result("logreg", yte, p_test, extra)


if __name__ == "__main__":
    main()
