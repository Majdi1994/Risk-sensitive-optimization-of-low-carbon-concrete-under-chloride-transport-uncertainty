"""Leave-one-source summary and dependence table from the files this run just wrote."""
from pathlib import Path
import json, csv, math
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- LOSO fold table ----
folds = pd.read_csv(OUT / "rev2_grouped_validation_folds.csv")
loso = pd.DataFrame({
    "source_id": folds["held_source_id"],
    "Reference": folds["held_source_label"],
    "n_held_out": folds["n_test"],
    "r2_log10Da": folds["after_r2"],
    "rmse_log10Da": folds["after_rmse"],
    "mae_log10Da": folds["after_mae"],
    "picp90_raw": folds["before_picp_90"],
    "picp90_lambda": folds["after_picp_90"],
    "is_dominant_source": folds["fold_type"].eq("dominant_source_holdout"),
})
loso_out = OUT / "loso_fold_metrics.csv"
loso.to_csv(loso_out, index=False)
dom = loso[loso["is_dominant_source"].astype(str).str.lower().isin(["true","1"])]
nondom = loso[~loso["is_dominant_source"].astype(str).str.lower().isin(["true","1"])]
# pooled approx weighted
def wmean(df, col):
    w = df["n_held_out"].astype(float)
    return float(np.average(df[col].astype(float), weights=w))
loso_summary = {
    "n_sources": int(len(loso)),
    "n_records": int(loso["n_held_out"].sum()),
    "pooled_r2_from_headline": 0.316,
    "pooled_rmse_from_headline": 0.350,
    "dominant_source_n": int(dom["n_held_out"].sum()) if len(dom) else None,
    "dominant_share": float(dom["n_held_out"].sum()/loso["n_held_out"].sum()) if len(dom) else None,
    "dominant_r2": float(dom["r2_log10Da"].iloc[0]) if len(dom) else None,
    "without_dominant_r2_from_loso24": json.loads(
        (OUT / "rev2_loso24_summary.json").read_text(encoding="utf-8")
    )["pooled_raw"]["r2"],
    "weighted_mean_r2_all": wmean(loso, "r2_log10Da"),
    "weighted_mean_rmse_all": wmean(loso, "rmse_log10Da"),
}
(OUT / "loso_summary.json").write_text(json.dumps(loso_summary, indent=2), encoding="utf-8")
print("LOSO summary", loso_summary)

# ---- Dependence one-pager ----
dep = json.loads((OUT / "rev2_dependence_sensitivity.json").read_text(encoding="utf-8"))
rows = []
for struct, payload in dep["structures"].items():
    b = payload["category_b_regret"]
    rows.append({
        "residual_structure": struct,
        "mean_pool_regret": b["mean_pool_mean"],
        "q10_pool_regret": b["q10_pool_mean"],
        "paired_delta_q10_minus_mean": b["delta_mean"],
        "ci95_low": b["paired_bootstrap_mean_delta_95ci"][0],
        "ci95_high": b["paired_bootstrap_mean_delta_95ci"][1],
        "relative_reduction_pct": 100*b["relative_mean_reduction"],
        "wilcoxon_p": b["paired_wilcoxon_two_sided_p"],
        "materially_reduces": b["materially_reduces"],
    })
pd.DataFrame(rows).to_csv(OUT / "dependence_regret_onepager.csv", index=False)
print("Dependence table written")

# ---- sigma vs |error| ----
preds = pd.read_csv(OUT / "rev2_grouped_validation_predictions.csv")
print("pred columns", list(preds.columns)[:30])
# try common names
cols = {c.lower(): c for c in preds.columns}
# inspect head
print(preds.head(2).to_dict())
