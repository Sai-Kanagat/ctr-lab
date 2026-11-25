"""Build dashboard/index.html from results/summary.json and dashboard/template.html.

Every numeric result on the page comes from results/summary.json (and, for the
logistic regression grid, results/logreg.json). Ranking claims typed in the prose
(DeepFM leads, LightGBM last and fastest, calibration dots mostly below the
diagonal) are asserted against those files. Setup details in the prose
(22 fields, 8 GB, M3, the 10% budget in headings) are typed into the template.
The data is inlined as a JSON script tag so the page works from file://.

Run:  .venv/bin/python dashboard/build.py
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "dashboard"

MODEL_ORDER = ["deepfm", "logreg", "lgbm"]
START_K = 10  # initial slider value, matches summary["headline"]["budget_pct"]


def n0(x):
    return f"{round(x):,}"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def esc(s):
    return html.escape(str(s), quote=True)


def main():
    summary = json.loads((RES / "summary.json").read_text())
    logreg = json.loads((RES / "logreg.json").read_text())

    h = summary["headline"]
    start_k = int(h["budget_pct"])
    assert start_k == START_K
    data = summary["data"]
    total = summary["test_total_clicks"]
    models = summary["models"]
    curves = summary["budget_curve"]

    grid = sorted(float(c) for c in logreg["per_C"])
    best_c = float(logreg["C"])
    grid_edge = best_c == grid[-1] or best_c == grid[0]
    grid_total_s = float(logreg["train_seconds_total_grid"])

    extra = {
        "logreg_C": best_c,
        "logreg_grid": grid,
        "logreg_grid_total_seconds": grid_total_s,
        "logreg_grid_edge": grid_edge,
    }
    payload = dict(summary)
    payload["extra"] = extra
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    # ---- headline sentence (static, so the page reads without JavaScript)
    lede = (
        f'Score the <span class="num">{n0(data["test"]["n"])}</span> impressions in the held-out test set with '
        f'{esc(h["best_model_label"])}, buy the top <span class="num">{n0(h["impressions_bought"])}</span>, '
        f'and you collect <span class="num">{n0(h["clicks_model"])}</span> of the '
        f'<span class="num">{n0(total)}</span> clicks '
        f'(<span class="num">{pct(h["share_of_clicks_model"])}</span>). '
        f'Buying the same number at random gets about <span class="num">{n0(h["clicks_random"])}</span>. '
        f'That is <span class="num">{h["lift_multiple"]:.2f}x</span> the clicks for the same number of impressions.'
    )

    # ---- slider table, initial state
    rows = []
    for key in MODEL_ORDER + ["random"]:
        pt = curves[key][start_k - 1]
        rnd = curves["random"][start_k - 1]["clicks"]
        label = "Random (expected)" if key == "random" else models[key]["label"]
        cls = ' class="best"' if key == h["best_model"] else ""
        rows.append(
            f'<tr{cls} data-key="{key}"><th scope="row">{esc(label)}</th>'
            f'<td class="num" data-f="clicks">{n0(pt["clicks"])}</td>'
            f'<td class="num" data-f="share">{pct(pt["share_of_clicks"])}</td>'
            f'<td class="num" data-f="lift">{pt["clicks"] / rnd:.2f}x</td></tr>'
        )
    slider_rows = "\n".join(rows)

    # ---- comparison table
    best_auc = max(models, key=lambda k: models[k]["auc"])
    best_ll = min(models, key=lambda k: models[k]["logloss"])
    best_clicks = max(models, key=lambda k: curves[k][start_k - 1]["clicks"])
    fastest = min(models, key=lambda k: models[k]["train_seconds"])
    cmp_rows = []
    for key in MODEL_ORDER:
        m = models[key]
        b = lambda ok: ' class="num lead"' if ok else ' class="num"'
        cmp_rows.append(
            f'<tr><th scope="row">{esc(m["label"])}</th>'
            f'<td{b(key == best_auc)}>{m["auc"]:.4f}</td>'
            f'<td{b(key == best_ll)}>{m["logloss"]:.4f}</td>'
            f'<td{b(key == best_ll)}>{m["logloss_gain_pct"]:.1f}%</td>'
            f'<td{b(key == best_clicks)}>{n0(curves[key][start_k - 1]["clicks"])}</td>'
            f'<td{b(key == fastest)}>{m["train_seconds"] / 60:.2f}</td></tr>'
        )
    cmp_rows = "\n".join(cmp_rows)
    const_ll = models[MODEL_ORDER[0]]["logloss_constant"]

    # Deliberate guards: the typed prose in template.html assumes this ranking.
    # If a re-run changes it, the build fails on purpose; edit the prose, then rebuild.
    assert best_auc == best_ll == h["best_model"] == "deepfm", "prose assumes DeepFM leads"
    assert best_c == grid[-1], "prose assumes best C is the top of the grid"
    assert min(models, key=lambda k: models[k]['logloss']) == "deepfm" and max(models, key=lambda k: models[k]['logloss']) == "lgbm", "prose assumes LightGBM is last on log loss"
    assert max(models, key=lambda k: models[k]['auc']) == best_auc
    assert min(models, key=lambda k: models[k]['auc']) == 'lgbm', "prose assumes LightGBM is last"
    assert best_clicks == "deepfm", "prose assumes DeepFM leads on clicks"
    assert fastest == "lgbm", "prose assumes LightGBM trains fastest"
    for mk, bins in summary["calibration"].items():
        below = sum(b["actual_ctr"] < b["mean_pred"] for b in bins if b["bin"] >= 2)
        assert below >= 8, f"prose assumes most dots below diagonal ({mk}: {below} of 9)"
    d_auc = models["deepfm"]["auc"] - models["logreg"]["auc"]

    tokens = {
        "DATA_JSON": blob,
        "LEDE": lede,
        "START_K": str(start_k),
        "SLIDER_ROWS": slider_rows,
        "CMP_ROWS": cmp_rows,
        "CONST_LL": f"{const_ll:.4f}",
        "N_TRAIN": n0(data["train"]["n"]),
        "N_VALID": n0(data["valid"]["n"]),
        "N_TEST": n0(data["test"]["n"]),
        "RATE_TRAIN": pct(data["train"]["click_rate"]),
        "RATE_VALID": pct(data["valid"]["click_rate"]),
        "RATE_TEST": pct(data["test"]["click_rate"]),
        "TOTAL_CLICKS": n0(total),
        "BEST_LABEL": esc(h["best_model_label"]),
        "LOGREG_C": f"{best_c:g}",
        "LOGREG_GRID": ", ".join(f"{g:g}" for g in grid),
        "LOGREG_GRID_MIN": f'{grid_total_s / 60:.0f}',
        "LOGREG_FIT_MIN": f'{models["logreg"]["train_seconds"] / 60:.0f}',
        "AUC_GAP": f"{d_auc:.4f}",
        "DEEPFM_AUC": f'{models["deepfm"]["auc"]:.4f}',
        "LOGREG_AUC": f'{models["logreg"]["auc"]:.4f}',
        "LGBM_AUC": f'{models["lgbm"]["auc"]:.4f}',
        "LGBM_CLICKS10": n0(curves["lgbm"][start_k - 1]["clicks"]),
        "LOGREG_CLICKS10": n0(curves["logreg"][start_k - 1]["clicks"]),
    }

    tpl = (OUT / "template.html").read_text()
    out = tpl
    for k, v in tokens.items():
        out = out.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
    if left:
        sys.exit(f"unfilled tokens: {sorted(set(left))}")

    # hard rule: no em or en dashes anywhere in the output
    for ch, name in ((chr(0x2014), "em dash"), (chr(0x2013), "en dash")):
        if ch in out:
            sys.exit(f"{name} found in output")

    (OUT / "index.html").write_text(out)
    print(f"wrote {OUT / 'index.html'} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
