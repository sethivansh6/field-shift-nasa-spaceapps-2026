"""
Step 6: train and test the crop-vigour model.

Run from the FARMS folder:
    python src/train_model.py

What it does (decisions 14 and 19-24 in PROJECT.md)
  1. Loads data/processed/training_sample.csv (region = train | test). Drops crop classes with < 100 rows
     and rows with missing features.
  2. For BOTH targets (ndvi_peak, ndvi_mean_mj), compares 5 models on ~100 km spatial block CV
     (train region only):
        crop_mean    crop average only (baseline)
        crop_site    LightGBM on crop + site_ndvi only (no weather)  -> what the NASA weather data adds
        linear       Ridge regression on all features
        mlp          neural network on all features
        lgbm         LightGBM on all features
     and scores every model on the held-out Lusatia test set.
  3. Picks the target where the full model adds most skill over crop_site on Lusatia.
  4. Year-transfer test: train 2018 -> test 2022 and the reverse.
  5. Quantile models (10th / 90th percentile) and their coverage on Lusatia.
  6. SHAP explanations (LightGBM) on Lusatia.
  7. Retrains the chosen model + quantiles on ALL data (train + Lusatia) and saves them.
     All reported scores come from before this retraining.
  8. Predicts every model crop for the 3 demo fields, 2015-2025 (used by the rotation scorer).

Outputs
  models/stress_model.joblib
  data/processed/model_report.json
  data/processed/demo_predictions.csv
  data/processed/figures/*.png
"""
import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PROCESSED, MODELS

warnings.filterwarnings("ignore")
FIG = DATA_PROCESSED / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = ["site_ndvi", "precip_spring", "precip_summer", "sm_rz_mj", "sm_rz_anom_mj", "pet_summer",
            "heat_days", "late_frost_days", "gdd", "sand", "clay", "soc", "ph", "bdod", "awc",
            "elevation", "slope", "tpi"]
WEATHER = ["precip_spring", "precip_summer", "sm_rz_mj", "sm_rz_anom_mj", "pet_summer",
           "heat_days", "late_frost_days", "gdd"]
TARGETS = ["ndvi_peak", "ndvi_mean_mj"]
MIN_ROWS_PER_CROP = 100
BLOCK_LON, BLOCK_LAT = 1.4, 0.9          # ~100 km x 100 km at 51 N
N_FOLDS = 5
SEED = 42


# ================================================================ data
def load():
    df = pd.read_csv(DATA_PROCESSED / "training_sample.csv")
    n0 = len(df)
    counts = df["crop_code"].value_counts()
    keep = counts[counts >= MIN_ROWS_PER_CROP].index
    dropped = {int(k): int(v) for k, v in counts[counts < MIN_ROWS_PER_CROP].items()}
    df = df[df["crop_code"].isin(keep)]
    df = df.dropna(subset=FEATURES + TARGETS)
    df["crop"] = df["crop_code"].astype(int).astype(str)
    df["block"] = ((df["lon"] // BLOCK_LON).astype(int).astype(str) + "_" +
                   (df["lat"] // BLOCK_LAT).astype(int).astype(str))
    info = {"rows_loaded": n0, "rows_used": len(df), "crop_classes_dropped": dropped,
            "crop_classes_used": sorted(int(c) for c in keep)}
    return df, info


# ================================================================ models
def lgbm(params=None, objective="regression", alpha=None):
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    p.update(params or {})
    if objective == "quantile":
        p.update(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(**p)


def prep(X):
    X = X.copy()
    X["crop"] = X["crop"].astype("category")
    return X


class CropMean:
    def fit(self, X, y):
        self.m = pd.Series(y.values, index=X["crop"].values).groupby(level=0).mean()
        self.g = float(np.mean(y))
        return self

    def predict(self, X):
        return X["crop"].map(self.m).fillna(self.g).values


class LGBMWrap:
    """LightGBM with the crop column as a categorical feature."""
    def __init__(self, cols, **kw):
        self.cols, self.kw = cols, kw

    def fit(self, X, y):
        self.cats = sorted(X["crop"].unique())
        Xc = X[self.cols].copy()
        Xc["crop"] = pd.Categorical(Xc["crop"], categories=self.cats)
        self.m = lgbm(**self.kw).fit(Xc, y)
        return self

    def predict(self, X):
        Xc = X[self.cols].copy()
        Xc["crop"] = pd.Categorical(Xc["crop"], categories=self.cats)
        return self.m.predict(Xc)


def sk_pipe(estimator):
    pre = ColumnTransformer([("num", StandardScaler(), FEATURES),
                             ("crop", OneHotEncoder(handle_unknown="ignore"), ["crop"])])
    return Pipeline([("pre", pre), ("est", estimator)])


def make(name, params=None):
    if name == "crop_mean":
        return CropMean()
    if name == "crop_site":
        return LGBMWrap(["crop", "site_ndvi"], params={"n_estimators": 300})
    if name == "linear":
        return sk_pipe(Ridge(alpha=1.0))
    if name == "mlp":
        p = {"hidden_layer_sizes": (64, 32), "alpha": 1e-3}
        p.update(params or {})
        return sk_pipe(MLPRegressor(early_stopping=True, max_iter=500, random_state=SEED, **p))
    if name == "lgbm":
        return LGBMWrap(["crop"] + FEATURES, params=params)
    raise ValueError(name)


def scores(y, p):
    return {"r2": round(float(r2_score(y, p)), 3), "mae": round(float(mean_absolute_error(y, p)), 4)}


def spatial_cv(name, df, target, params=None):
    tr = df[df["region"] == "train"].reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    r2s, maes = [], []
    for a, b in gkf.split(tr, groups=tr["block"]):
        m = make(name, params).fit(tr.loc[a, ["crop"] + FEATURES], tr.loc[a, target])
        p = m.predict(tr.loc[b, ["crop"] + FEATURES])
        r2s.append(r2_score(tr.loc[b, target], p))
        maes.append(mean_absolute_error(tr.loc[b, target], p))
    return {"r2": round(float(np.mean(r2s)), 3), "r2_sd": round(float(np.std(r2s)), 3),
            "mae": round(float(np.mean(maes)), 4)}


def lusatia_score(name, df, target, params=None):
    tr, te = df[df["region"] == "train"], df[df["region"] == "test"]
    m = make(name, params).fit(tr[["crop"] + FEATURES], tr[target])
    return scores(te[target], m.predict(te[["crop"] + FEATURES]))


# ================================================================ main steps
TUNE = {"lgbm": [{"num_leaves": 15}, {"num_leaves": 31}, {"num_leaves": 63}],
        "mlp": [{"hidden_layer_sizes": (64, 32)}, {"hidden_layer_sizes": (128, 64)}]}


def compare(df):
    out = {}
    for target in TARGETS:
        print(f"\n=== target: {target}")
        res = {}
        for name in ["crop_mean", "crop_site", "linear", "mlp", "lgbm"]:
            options = TUNE.get(name, [None])
            best = None
            for params in options:
                cv = spatial_cv(name, df, target, params)
                if best is None or cv["r2"] > best[1]["r2"]:
                    best = (params, cv)
            params, cv = best
            lus = lusatia_score(name, df, target, params)
            res[name] = {"params": params, "spatial_cv": cv, "lusatia": lus}
            print(f"  {name:10s} CV R2 {cv['r2']:6.3f} (+-{cv['r2_sd']:.3f})  MAE {cv['mae']:.4f}   "
                  f"Lusatia R2 {lus['r2']:6.3f}  MAE {lus['mae']:.4f}   {params or ''}")
        out[target] = res
    return out


def year_transfer(df, target, name, params):
    tr = df[df["region"] == "train"]
    res = {}
    for a, b in [(2018, 2022), (2022, 2018)]:
        A, B = tr[tr["year"] == a], tr[tr["year"] == b]
        m = make(name, params).fit(A[["crop"] + FEATURES], A[target])
        base = make("crop_mean").fit(A[["crop"] + FEATURES], A[target])
        res[f"{a}->{b}"] = {"model": scores(B[target], m.predict(B[["crop"] + FEATURES])),
                            "crop_mean": scores(B[target], base.predict(B[["crop"] + FEATURES]))}
    return res


def fit_q(X, y, params, alpha):
    return LGBMWrap(["crop"] + FEATURES, params=params, objective="quantile", alpha=alpha).fit(X, y)


def conformal_margin(df, target, params, coverage=0.8):
    """Split-conformal calibration of the 10-90 % range (conformalized quantile regression).
    Out-of-fold quantile predictions on spatial CV give conformity scores E = max(q10 - y, y - q90);
    the margin Q is their (coverage)-quantile. Final range = [q10 - Q, q90 + Q]."""
    tr = df[df["region"] == "train"].reset_index(drop=True)
    E = np.zeros(len(tr))
    for a, b in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr["block"]):
        X, y = tr.loc[a, ["crop"] + FEATURES], tr.loc[a, target]
        lo = fit_q(X, y, params, 0.1).predict(tr.loc[b])
        hi = fit_q(X, y, params, 0.9).predict(tr.loc[b])
        E[b] = np.maximum(lo - tr.loc[b, target].values, tr.loc[b, target].values - hi)
    n = len(E)
    return float(np.quantile(E, min(1.0, np.ceil((n + 1) * coverage) / n)))


def quantiles(df, target, params):
    tr, te = df[df["region"] == "train"], df[df["region"] == "test"]
    X, y = tr[["crop"] + FEATURES], tr[target]
    lo, hi = fit_q(X, y, params, 0.1).predict(te), fit_q(X, y, params, 0.9).predict(te)
    raw = ((te[target] >= lo) & (te[target] <= hi)).mean()
    Q = conformal_margin(df, target, params)
    cal = ((te[target] >= lo - Q) & (te[target] <= hi + Q)).mean()
    return {"raw_coverage": round(float(raw), 3), "raw_width": round(float(np.mean(hi - lo)), 4),
            "margin": round(Q, 4), "calibrated_coverage": round(float(cal), 3),
            "calibrated_width": round(float(np.mean(hi - lo + 2 * Q)), 4)}


def shap_explain(df, target, params, crop_names):
    tr, te = df[df["region"] == "train"], df[df["region"] == "test"]
    m = LGBMWrap(["crop"] + FEATURES, params=params).fit(tr[["crop"] + FEATURES], tr[target])
    Xte = te[["crop"] + FEATURES].copy()
    Xte["crop"] = pd.Categorical(Xte["crop"], categories=m.cats)
    sv = shap.TreeExplainer(m.m).shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xte.columns).sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    imp.head(12)[::-1].plot.barh(color="#3B6D11")
    plt.xlabel("mean |SHAP| (effect on predicted NDVI)")
    plt.title(f"What drives crop vigour in Lusatia ({target})")
    plt.tight_layout()
    plt.savefig(FIG / "shap_importance.png", dpi=150)
    plt.close()

    for feat in ("sm_rz_anom_mj", "heat_days", "precip_summer"):
        i = list(Xte.columns).index(feat)
        plt.figure(figsize=(6, 4))
        plt.scatter(Xte[feat], sv[:, i], s=6, alpha=0.5, color="#185FA5")
        plt.axhline(0, color="grey", lw=0.5)
        plt.xlabel(feat)
        plt.ylabel("SHAP effect on NDVI")
        plt.title(f"Effect of {feat} (Lusatia test points)")
        plt.tight_layout()
        plt.savefig(FIG / f"shap_{feat}.png", dpi=150)
        plt.close()
    return [str(x) for x in imp.head(5).index], {k: round(float(v), 4) for k, v in imp.items()}


def drought_check(df, target, name, params):
    """Does the model see 2018 as worse than 2022 in Lusatia, like the observations?"""
    tr, te = df[df["region"] == "train"], df[df["region"] == "test"]
    m = make(name, params).fit(tr[["crop"] + FEATURES], tr[target])
    te = te.assign(pred=m.predict(te[["crop"] + FEATURES]))
    g = te.groupby("year")[[target, "pred"]].mean().round(3)
    return {int(y): {"observed": float(r[target]), "predicted": float(r["pred"])} for y, r in g.iterrows()}


def demo_predictions(model, q10, q90, crops, margin):
    demo = pd.read_csv(DATA_PROCESSED / "demo_season_features.csv")
    rows = []
    for c in crops:
        d = demo.copy()
        d["crop"] = str(c)
        d["crop_code"] = c
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).dropna(subset=FEATURES)
    d["pred"] = model.predict(d[["crop"] + FEATURES])
    d["pred_q10"] = q10.predict(d[["crop"] + FEATURES]) - margin   # conformal-calibrated
    d["pred_q90"] = q90.predict(d[["crop"] + FEATURES]) + margin
    keep = ["field_id", "year", "crop_code", "pred", "pred_q10", "pred_q90"] + \
           [c for c in ("ndvi_peak", "ndvi_mean_mj") if c in d] + WEATHER
    out = d[keep].sort_values(["field_id", "crop_code", "year"])
    out.to_csv(DATA_PROCESSED / "demo_predictions.csv", index=False)
    return out


def main():
    df, info = load()
    legend_p = DATA_PROCESSED.parent / "raw" / "gee" / "eucropmap_legend.json"
    legend = {int(k): v for k, v in json.loads(legend_p.read_text()).items()} if legend_p.exists() else {}
    print(f"Rows used: {info['rows_used']} of {info['rows_loaded']}  "
          f"(train {sum(df.region == 'train')}, Lusatia test {sum(df.region == 'test')})")
    print(f"Crop classes used: {[legend.get(c, c) for c in info['crop_classes_used']]}")
    print(f"Dropped (< {MIN_ROWS_PER_CROP} rows): {info['crop_classes_dropped']}")
    print(f"Spatial blocks: {df['block'].nunique()} (~100 km)")

    res = compare(df)

    # choose target: most skill added by NASA weather data on Lusatia (lgbm minus crop_site)
    gain = {t: res[t]["lgbm"]["lusatia"]["r2"] - res[t]["crop_site"]["lusatia"]["r2"] for t in TARGETS}
    target = max(gain, key=gain.get)
    # choose model type on spatial CV for that target
    cands = ["linear", "mlp", "lgbm"]
    best = max(cands, key=lambda n: res[target][n]["spatial_cv"]["r2"])
    params = res[target][best]["params"]
    lgbm_params = res[target]["lgbm"]["params"]
    print(f"\nChosen target: {target} (weather adds {gain[target]:+.3f} R2 on Lusatia; "
          f"other target {', '.join(f'{t} {g:+.3f}' for t, g in gain.items() if t != target)})")
    print(f"Chosen model: {best} {params or ''}")

    yt = year_transfer(df, target, best, params)
    print("\nYear transfer (train one year, test the other):")
    for k, v in yt.items():
        print(f"  {k}: model R2 {v['model']['r2']:.3f}  vs crop mean R2 {v['crop_mean']['r2']:.3f}")

    qr = quantiles(df, target, lgbm_params)
    print(f"\n10-90% range on Lusatia: raw coverage {qr['raw_coverage']:.0%} (width {qr['raw_width']:.3f}) -> "
          f"calibrated {qr['calibrated_coverage']:.0%} (width {qr['calibrated_width']:.3f}, margin +-{qr['margin']:.3f})")

    top5, imp = shap_explain(df, target, lgbm_params, legend)
    print(f"SHAP top 5 (LightGBM, Lusatia): {top5}")

    dc = drought_check(df, target, best, params)
    print(f"Lusatia mean {target} by year (observed vs predicted): {dc}")

    # ------------------------------------------------ final models on ALL data
    final = make(best, params).fit(df[["crop"] + FEATURES], df[target])
    q10 = LGBMWrap(["crop"] + FEATURES, params=lgbm_params, objective="quantile", alpha=0.1) \
        .fit(df[["crop"] + FEATURES], df[target])
    q90 = LGBMWrap(["crop"] + FEATURES, params=lgbm_params, objective="quantile", alpha=0.9) \
        .fit(df[["crop"] + FEATURES], df[target])
    crops = info["crop_classes_used"]
    joblib.dump({"model": final, "q10": q10, "q90": q90, "model_type": best, "params": params,
                 "target": target, "features": FEATURES, "crop_codes": crops,
                 "crop_names": {c: legend.get(c) for c in crops}, "conformal_margin": qr["margin"]},
                MODELS / "stress_model.joblib")

    preds = demo_predictions(final, q10, q90, crops, qr["margin"])
    obs_col = target if target in preds else None
    if obs_col:
        cm = pd.read_csv(DATA_PROCESSED.parent / "raw" / "gee" / "eucropmap.csv")
        chk = preds.merge(cm[["field_id", "year", "crop_code"]], on=["field_id", "year", "crop_code"])
        if len(chk):
            print("\nDemo fields, actual crop, observed vs predicted:")
            print(chk[["field_id", "year", "crop_code", obs_col, "pred", "pred_q10", "pred_q90"]]
                  .round(3).to_string(index=False))

    report = {
        "data": info, "blocks": int(df["block"].nunique()),
        "comparison": res, "target_gain_over_crop_site": {k: round(v, 3) for k, v in gain.items()},
        "chosen_target": target, "chosen_model": best, "chosen_params": params,
        "cv_r2": res[target][best]["spatial_cv"]["r2"],
        "baseline_r2": res[target]["crop_mean"]["spatial_cv"]["r2"],
        "crop_site_r2": res[target]["crop_site"]["spatial_cv"]["r2"],
        "lusatia_r2": res[target][best]["lusatia"]["r2"],
        "year_transfer": yt, "quantile_coverage_lusatia": qr["calibrated_coverage"], "quantiles": qr,
        "shap_top5": top5, "shap_importance": imp, "lusatia_drought_check": dc,
        "note": "Scores are from models trained before the final retraining on all data (train + Lusatia).",
    }
    (DATA_PROCESSED / "model_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved models/stress_model.joblib, model_report.json, demo_predictions.csv ({len(preds)} rows), figures/")


if __name__ == "__main__":
    main()