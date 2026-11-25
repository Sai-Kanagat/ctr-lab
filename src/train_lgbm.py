"""LightGBM CTR model. Categorical indices (<=1000 distinct) + train-fit frequency encoding for every field.
Early stopping on VALID logloss only; TEST is touched once at the end for the final prediction."""
import argparse, sys, time
from pathlib import Path
import numpy as np, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, build_vocab, encode, save_result, FEATURES

SEED = 42
MAX_CARD = 1000


def count_encode(df, counts):
    """Frequency of each raw id in TRAIN (0 for ids never seen in train)."""
    return np.stack([df[c].map(counts[c]).fillna(0).to_numpy(dtype="float32") for c in FEATURES], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    np.random.seed(SEED)

    tr, va, te = (load(s, args.limit) for s in ("train", "valid", "test"))
    vocab = build_vocab(tr)
    counts = {c: tr[c].value_counts() for c in FEATURES}

    # literal spec rule: categorical iff the raw train field has <= MAX_CARD distinct values
    cat_idx = [i for i, c in enumerate(FEATURES) if counts[c].size <= MAX_CARD]

    def build(df):
        enc = encode(df, vocab)[:, cat_idx].astype("int32")
        cnt = count_encode(df, counts)
        X = np.hstack([enc.astype("float32"), cnt])
        return X, df["label"].to_numpy()

    names = [f"cat_{FEATURES[i]}" for i in cat_idx] + [f"cnt_{c}" for c in FEATURES]
    cat_cols = list(range(len(cat_idx)))
    Xtr, ytr = build(tr)
    Xva, yva = build(va)
    del tr, va

    params = dict(objective="binary", metric="binary_logloss", num_leaves=63, learning_rate=0.1,
                  num_threads=4, seed=SEED, deterministic=True, force_row_wise=True, verbose=-1,
                  cat_smooth=10, min_data_per_group=100, max_cat_to_onehot=4)
    t0 = time.time()
    dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    dtr.construct(); dva.construct()  # binning counted in train_seconds

    booster = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
    train_seconds = time.time() - t0
    best = int(booster.best_iteration or booster.current_iteration())

    gain = booster.feature_importance(importance_type="gain", iteration=best)
    top = sorted(zip(names, gain.tolist()), key=lambda t: -t[1])[:8]
    extra = {"best_iteration": best, "train_seconds": round(train_seconds, 2),
             "top_features": [[n, round(g, 2)] for n, g in top]}

    # TEST touched exactly once, here
    Xte, yte = build(te)
    p_test = booster.predict(Xte, num_iteration=best)
    save_result("lgbm", yte, p_test, extra)


if __name__ == "__main__":
    main()
