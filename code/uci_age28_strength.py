"""Reproduce the saved UCI GBR cell and print the missing train R2."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
xlsx = ROOT / "data" / "Compressive_Strength.xlsx"
df = pd.read_excel(xlsx)
df.columns = ["Cement", "Slag", "FA", "Water", "SP", "NCA", "NFA", "Age", "CS"]
y = df["CS"].values
X = df.drop(columns=["CS"]).copy()
X_fe = X.copy()
X_fe["w_c"] = X_fe["Water"] / (X_fe["Cement"] + 1e-9)
X_fe["Binder"] = X_fe[["Cement", "Slag", "FA"]].sum(axis=1)
X_fe["w_b"] = X_fe["Water"] / (X_fe["Binder"] + 1e-9)
X_fe["SCM_frac"] = X_fe[["Slag", "FA"]].sum(axis=1) / (X_fe["Binder"] + 1e-9)
X_fe["Agg_total"] = X_fe["NCA"] + X_fe["NFA"]
X_fe["Agg_to_Binder"] = X_fe["Agg_total"] / (X_fe["Binder"] + 1e-9)
X_fe["SP_per_binder"] = X_fe["SP"] / (X_fe["Binder"] + 1e-9)
X_fe["Water_SP"] = X_fe["Water"] * X_fe["SP"]
X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y, test_size=0.2, random_state=42
)
gbr = GradientBoostingRegressor(
    subsample=0.8,
    n_estimators=400,
    min_samples_leaf=3,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)
gbr.fit(X_train, y_train)
pred_tr = gbr.predict(X_train)
pred_te = gbr.predict(X_test)
out = {
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "train_r2": float(r2_score(y_train, pred_tr)),
    "test_r2": float(r2_score(y_test, pred_te)),
    "test_rmse": float(mean_squared_error(y_test, pred_te) ** 0.5),
    "test_mae": float(mean_absolute_error(y_test, pred_te)),
}
print(out)
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "uci_train_r2.json").write_text(
    __import__("json").dumps(out, indent=2), encoding="utf-8"
)
