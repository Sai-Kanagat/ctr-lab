"""Score the three CTR models on the held-out test split and write results/summary.json.

All numbers come from data/*.parquet labels and results/*.json / preds_*.npy.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = {"logreg": "Logistic regression", "lgbm": "LightGBM", "deepfm": "DeepFM"}
INK, ACCENT, GREY = "#1a1a1a", "#d9480f", "#8a8a8a"


def load_labels(split):
    return pd.read_parquet(ROOT / "data" / f"{split}.parquet", columns=["label"])["label"].to_numpy()


def budget_curve(y, pred):
    """Clicks captured when buying the top k% of impressions by predicted CTR, k=1..100."""
    n, total = len(y), int(y.sum())
    cum = np.cumsum(y[np.argsort(-pred, kind="stable")])
    rows = []
    for k in range(1, 101):
        clicks = int(cum[round(n * k / 100) - 1])
        rows.append({"k_pct": k, "clicks": clicks, "share_of_clicks": clicks / total})
    return rows


def random_curve(y):
    n, total = len(y), int(y.sum())
    return [{"k_pct": k, "clicks": total * k / 100, "share_of_clicks": k / 100} for k in range(1, 101)]


def calibration(y, pred, bins=10):
    order = np.argsort(pred, kind="stable")
    return [{"bin": i + 1, "mean_pred": float(pred[idx].mean()), "actual_ctr": float(y[idx].mean()), "n": int(len(idx))}
            for i, idx in enumerate(np.array_split(order, bins))]


def main():
    y = load_labels("test")
    n = len(y)
    splits = {s: load_labels(s) for s in ("train", "valid")}
    splits["test"] = y
    data = {s: {"n": int(len(v)), "clicks": int(v.sum()), "click_rate": float(v.mean())} for s, v in splits.items()}

    models, curves, calib, preds = {}, {}, {}, {}
    for key, label in MODELS.items():
        r = json.load(open(RES / f"{key}.json"))
        p = np.load(RES / f"preds_{key}.npy").astype(np.float64)
        assert len(p) == n and r["n"] == n, f"{key}: prediction length mismatch"
        preds[key] = p
        models[key] = {"name": key, "label": label, "auc": r["auc"], "logloss": r["logloss"],
                       "logloss_constant": r["logloss_constant"],
                       "logloss_gain_pct": 100 * (r["logloss_constant"] - r["logloss"]) / r["logloss_constant"],
                       "train_seconds": r["train_seconds"]}
        curves[key] = budget_curve(y, p)
        calib[key] = calibration(y, p)
    curves["random"] = random_curve(y)

    # Headline: buy 10% of impressions. Best model = most clicks captured at that budget.
    at10 = lambda key: curves[key][9]
    best = max(MODELS, key=lambda k: at10(k)["clicks"])
    n_bought = round(n * 0.10)
    m, rnd = at10(best)["clicks"], at10("random")["clicks"]
    headline = {"budget_pct": 10, "impressions_bought": n_bought, "best_model": best,
                "best_model_label": MODELS[best],
                "clicks_model": m, "clicks_random": rnd, "lift_multiple": m / rnd,
                "clicks_per_1000_model": 1000 * m / n_bought, "clicks_per_1000_random": 1000 * rnd / n_bought,
                "share_of_clicks_model": at10(best)["share_of_clicks"]}

    summary = {"models": models, "budget_curve": curves, "headline": headline, "calibration": calib,
               "data": data, "test_total_clicks": int(y.sum())}
    json.dump(summary, open(RES / "summary.json", "w"), indent=2)

    # Plot: ink lines, one accent for the best model, random as dashed reference.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    x = np.arange(1, 101)
    ax.plot(x, [r["share_of_clicks"] * 100 for r in curves["random"]], "--", color=GREY, lw=1, label="Random")
    styles = {"logreg": (INK, ":"), "lgbm": (INK, "-."), "deepfm": (INK, "-")}
    for key in MODELS:
        color, ls = (ACCENT, "-") if key == best else styles[key]
        ax.plot(x, [r["share_of_clicks"] * 100 for r in curves[key]], ls, color=color,
                lw=2.2 if key == best else 1.3, label=MODELS[key])
    ax.set_xlabel("Impressions bought (%, highest predicted CTR first)")
    ax.set_ylabel("Share of all clicks captured (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(alpha=0.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "budget_curve.png")

    print(json.dumps({"models": models, "headline": headline, "data": data}, indent=2))


if __name__ == "__main__":
    main()
