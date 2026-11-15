"""
Automated validation checks for every FARMS step (see PROJECT.md section 6).

Usage (from the FARMS folder):
    python src/validate.py --step 1
    python src/validate.py --step 2
    python src/validate.py --all        # run every step whose files exist

Exit code 0 = all checks passed, 1 = at least one failed.
"""
import argparse
import json
import sys

import pandas as pd

from config import (FIELDS, RAW_GEE, RAW_POWER, RAW_ECOSTRESS, DATA_PROCESSED, MODELS, ROOT)

FIELD_IDS = {f["id"] for f in FIELDS}


class Checker:
    def __init__(self, step):
        self.step = step
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))

    def report(self):
        print(f"\n=== Step {self.step} ===")
        for name, ok, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
        n_fail = sum(not ok for _, ok, _ in self.results)
        print(f"  -> {len(self.results) - n_fail}/{len(self.results)} passed")
        return n_fail == 0


def read_csv(path):
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------- step 1
def step1():
    c = Checker(1)
    names = ["smap_monthly", "imerg_monthly", "modis_ndvi", "modis_lst", "modis_et",
             "grace_mascon", "terrain", "soilgrids", "eucropmap"]
    dfs = {n: read_csv(RAW_GEE / f"{n}.csv") for n in names}
    missing = [n for n, d in dfs.items() if d is None]
    c.check("all 9 GEE files exist", not missing, f"missing: {missing}" if missing else "")
    if missing:
        return c

    min_rows = {"smap_monthly": 350, "imerg_monthly": 800, "modis_ndvi": 1500,
                "modis_lst": 3000, "modis_et": 3000, "grace_mascon": 500}
    for n, m in min_rows.items():
        d = dfs[n]
        c.check(f"{n}: >= {m} rows", len(d) >= m, f"{len(d)} rows")
        c.check(f"{n}: all 3 fields", set(d["field_id"]) == FIELD_IDS)

    ndvi = dfs["modis_ndvi"]["NDVI"].dropna()
    c.check("NDVI within -0.2..1", ndvi.between(-0.2, 1).all(), f"{ndvi.min():.2f}..{ndvi.max():.2f}")
    sm = dfs["smap_monthly"]["sm_rootzone"].dropna()
    c.check("SMAP sm_rootzone within 0..0.9", sm.between(0, 0.9).all(), f"{sm.min():.2f}..{sm.max():.2f}")
    lst = dfs["modis_lst"]["LST_Day_1km"].dropna()
    c.check("LST day within -40..70 C", lst.between(-40, 70).all(), f"{lst.min():.1f}..{lst.max():.1f}")
    et = dfs["modis_et"]["ET"].dropna()
    c.check("ET >= 0", (et >= 0).all(), f"min {et.min():.2f}")

    crop = dfs["eucropmap"]
    c.check(f"eucropmap: {2*len(FIELD_IDS)} rows", len(crop) == 2 * len(FIELD_IDS), f"{len(crop)} rows")
    c.check("eucropmap: all arable codes 200..299",
            crop["crop_code"].between(200, 299).all(), str(crop["crop_code"].tolist()))
    c.check("soilgrids and terrain: one row per field",
            len(dfs["soilgrids"]) == len(FIELD_IDS) and len(dfs["terrain"]) == len(FIELD_IDS))
    return c


# ---------------------------------------------------------------- step 2
def step2():
    c = Checker(2)
    d = read_csv(RAW_POWER / "power_daily.csv")
    c.check("power_daily.csv exists", d is not None)
    if d is None:
        return c
    d["date"] = pd.to_datetime(d["date"])

    counts = d.groupby("field_id").size()
    c.check(f"{len(FIELD_IDS)} fields x >= 16,000 days",
            set(counts.index) == FIELD_IDS and (counts >= 16000).all(), counts.to_dict())
    c.check("no duplicate (field, date)", not d.duplicated(["field_id", "date"]).any())

    order = (d["T2M_MIN"] <= d["T2M"] + 0.05) & (d["T2M"] <= d["T2M_MAX"] + 0.05)
    c.check("T2M_MIN <= T2M <= T2M_MAX on >= 99% of days", order.mean() >= 0.99, f"{order.mean():.4f}")
    c.check("precipitation >= 0", (d["PRECTOTCORR"].dropna() >= 0).all())

    full = d[(d["date"].dt.year >= 1991) & (d["date"].dt.year <= 2020)]
    rain_year = full.groupby(["field_id", full["date"].dt.year])["PRECTOTCORR"].sum().groupby("field_id").mean()
    temp = full.groupby("field_id")["T2M"].mean()
    c.check("mean annual rain 400..800 mm (1991-2020)", rain_year.between(400, 800).all(),
            rain_year.round(0).to_dict())
    c.check("mean temperature 7..12 C (1991-2020)", temp.between(7, 12).all(), temp.round(1).to_dict())

    season = d[d["date"].dt.month.between(4, 8)]
    by_year = season.groupby(["field_id", season["date"].dt.year])["PRECTOTCORR"].sum()
    normal = by_year[by_year.index.get_level_values(1).isin(range(1991, 2021))].groupby("field_id").mean()
    y2018 = by_year.xs(2018, level=1)
    c.check("2018 Apr-Aug rain below 1991-2020 average", (y2018 < normal).all(),
            f"2018 {y2018.round(0).to_dict()} vs normal {normal.round(0).to_dict()}")
    return c


# ---------------------------------------------------------------- step 3
def step3():
    c = Checker(3)
    task = RAW_ECOSTRESS / "appeears_task.json"
    csvs = list(RAW_ECOSTRESS.glob("*.csv"))
    c.check("AppEEARS task recorded", task.exists())
    c.check("ECOSTRESS CSVs downloaded", len(csvs) > 0, f"{len(csvs)} files")
    for key in ("ESI", "JET", "LSTE", "SM"):
        c.check(f"a CSV for {key}", any(key in p.name.upper() for p in csvs))

    clean = read_csv(DATA_PROCESSED / "ecostress_clean.csv")
    c.check("ecostress_clean.csv exists (run load_ecostress.py)", clean is not None)
    if clean is not None and "esi" in clean:
        c.check("all 3 fields present", set(clean["field_id"]) == FIELD_IDS)
        n = clean.dropna(subset=["esi"]).groupby("field_id").size()
        c.check(">= 20 valid ESI obs per field", (n >= 20).all() and len(n) == 3, n.to_dict())
        esi = clean["esi"].dropna()
        c.check("ESI within 0..1.5", esi.between(0, 1.5).all(), f"{esi.min():.2f}..{esi.max():.2f}")
    return c


# ---------------------------------------------------------------- step 4
FEATURES = ["site_ndvi", "precip_spring", "precip_summer", "sm_rz_mj", "sm_rz_anom_mj", "pet_summer",
            "heat_days", "late_frost_days", "gdd", "sand", "clay", "soc", "ph", "bdod",
            "awc", "elevation", "slope", "tpi"]


def step4():
    from config import TEST_BOX, TRAIN_BOX
    c = Checker(4)
    t = read_csv(DATA_PROCESSED / "training_sample.csv")
    c.check("training_sample.csv exists", t is not None)
    if t is not None:
        test, train = t[t["region"] == "test"], t[t["region"] == "train"]
        c.check("years only 2018 and 2022", set(t["year"]) <= {2018, 2022}, str(sorted(set(t["year"]))))
        tol = 0.01   # sample points sit at 250 m pixel centres, which can fall just past the box edge
        w, s_, e, n = TEST_BOX[0] - tol, TEST_BOX[1] - tol, TEST_BOX[2] + tol, TEST_BOX[3] + tol
        c.check("Lusatia test set: >= 500 rows, all inside TEST_BOX (+-0.01 deg)",
                len(test) >= 500 and test["lon"].between(w, e).all() and test["lat"].between(s_, n).all(),
                f"{len(test)} rows")
        if len(train):
            tw, ts, te, tn = TEST_BOX
            inside = train["lon"].between(tw, te) & train["lat"].between(ts, tn)
            c.check("train set: >= 5,000 rows", len(train) >= 5000, f"{len(train)} rows")
            c.check("no train point inside Lusatia (held out)", not inside.any(), f"{inside.sum()} inside")
            W, S, E, N = TRAIN_BOX
            c.check("train points inside TRAIN_BOX (+-0.01 deg)",
                    train["lon"].between(W - tol, E + tol).all() and train["lat"].between(S - tol, N + tol).all())
        else:
            c.check("train set present (skip if you only ran --only test)", False, "0 rows")
        per_crop = t["crop_code"].value_counts()
        c.check(">= 6 crop classes with >= 100 rows", (per_crop >= 100).sum() >= 6, per_crop.head(12).to_dict())
        tgt = t["ndvi_peak"]
        c.check("ndvi_peak present in >= 95% rows", tgt.notna().mean() >= 0.95, f"{tgt.notna().mean():.3f}")
        c.check("ndvi_peak within 0.2..1.0 for >= 98%", tgt.dropna().between(0.2, 1.0).mean() >= 0.98,
                f"{tgt.dropna().between(0.2, 1.0).mean():.3f}")
        missing = sorted(set(FEATURES) - set(t.columns))
        c.check("all feature columns present", not missing, f"missing: {missing}" if missing else "")
        cols = [f for f in FEATURES if f in t]
        complete = t[cols].notna().all(axis=1).mean() if cols else 0
        c.check("features complete in >= 95% rows", complete >= 0.95, f"{complete:.3f}")
        if "heat_days" in t:
            hot = (t.loc[t["year"] == 2018, "heat_days"] > 0).mean()
            c.check("heat_days > 0 for most 2018 points", hot >= 0.5, f"{hot:.2f}")
        if "awc" in t:
            c.check("awc within 3..35 % for >= 95%", t["awc"].dropna().between(3, 35).mean() >= 0.95,
                    f"{t['awc'].min():.1f}..{t['awc'].max():.1f}")

    demo = read_csv(DATA_PROCESSED / "demo_season_features.csv")
    c.check("demo_season_features.csv exists", demo is not None)
    if demo is not None:
        c.check(f"{len(FIELD_IDS)} fields x 11 seasons (2015..2025)",
                len(demo) == 11 * len(FIELD_IDS) and set(demo["year"]) == set(range(2015, 2026)), f"{len(demo)} rows")
        missing = sorted(set(FEATURES) - set(demo.columns))
        c.check("demo has all feature columns", not missing, f"missing: {missing}" if missing else "")
        if not missing:
            comp = demo[FEATURES].notna().all(axis=1).mean()
            c.check("demo features complete in >= 90% rows", comp >= 0.9, f"{comp:.2f}")
    return c


# ---------------------------------------------------------------- steps 5-9 (file presence for now)
def step5():
    c = Checker(5)
    p = DATA_PROCESSED / "field_profiles.json"
    c.check("field_profiles.json exists", p.exists())
    if p.exists():
        prof = json.loads(p.read_text())
        c.check("profile for each field", FIELD_IDS <= set(prof), str(sorted(prof)))
        for fid in sorted(FIELD_IDS & set(prof)):
            pr = prof[fid]
            doy = pr["frost"]["last_spring_median_doy"]
            c.check(f"{fid}: last spring frost median 15 Mar..15 May", doy is not None and 74 <= doy <= 135,
                    pr["frost"]["last_spring_median"])
            awc = pr["soil_terrain"]["awc_pct_volume"]
            c.check(f"{fid}: water holding capacity 5..30 %", awc is not None and 5 <= awc <= 30, str(awc))
            et0 = pr["water"]["et0_year_normal_mm"]
            c.check(f"{fid}: annual ET0 500..800 mm", et0 is not None and 500 <= et0 <= 800, str(et0))
            rain = pr["water"]["rain_year_normal_mm"]
            c.check(f"{fid}: annual rain 400..800 mm", rain is not None and 400 <= rain <= 800, str(rain))
    s2 = read_csv(RAW_GEE / "s2_ndvi.csv")
    c.check("s2_ndvi.csv exists (run extract_s2.py)", s2 is not None)
    if s2 is not None:
        n = s2.groupby("field_id").size()
        c.check("Sentinel-2: >= 100 clear images per field", set(n.index) == FIELD_IDS and (n >= 100).all(),
                n.to_dict())
        c.check("Sentinel-2 NDVI within -0.2..1", s2["ndvi_median"].between(-0.2, 1).all())
    if p.exists():
        bm = [prof[f].get("bare_soil_sentinel2", {}) or {} for f in FIELD_IDS & set(prof)]
        vals = [b.get("bare_months_per_year") for b in bm if b.get("bare_months_per_year") is not None]
        c.check("Sentinel-2 bare months per year within 0..8, at least one field > 0.5",
                bool(vals) and all(0 <= v <= 8 for v in vals) and max(vals) > 0.5, str(vals))
    m = read_csv(DATA_PROCESSED / "monthly_field_table.csv")
    c.check(f"monthly_field_table.csv: {len(FIELD_IDS)} fields x 12 months", m is not None and len(m) == 12 * len(FIELD_IDS),
            f"{0 if m is None else len(m)} rows")
    y = read_csv(DATA_PROCESSED / "yearly_field_table.csv")
    c.check(f"yearly_field_table.csv: {len(FIELD_IDS)} fields, years 1981..2025", y is not None and
            set(y["field_id"]) == FIELD_IDS and y["year"].min() <= 1981 and y["year"].max() >= 2025)
    return c


def step6():
    c = Checker(6)
    c.check("models/stress_model.joblib exists", (MODELS / "stress_model.joblib").exists())
    rp = DATA_PROCESSED / "model_report.json"
    c.check("model_report.json exists", rp.exists())
    if rp.exists():
        r = json.loads(rp.read_text())
        cv, base, site = r.get("cv_r2"), r.get("baseline_r2"), r.get("crop_site_r2")
        c.check("4+ models compared (crop_mean, crop_site, linear, mlp, lgbm)",
                all(m in r["comparison"][r["chosen_target"]] for m in ("crop_mean", "crop_site", "linear", "mlp", "lgbm")))
        c.check("chosen model: spatial CV R2 > 0", cv is not None and cv > 0, f"cv_r2={cv}")
        c.check("beats crop-average baseline", cv is not None and base is not None and cv > base,
                f"cv={cv}, crop_mean={base}")
        c.check("beats crop + field quality (weather data adds skill)", cv is not None and site is not None and cv > site,
                f"cv={cv}, crop_site={site}")
        c.check("Lusatia held-out R2 > 0", (r.get("lusatia_r2") or -1) > 0, str(r.get("lusatia_r2")))
        yt = r.get("year_transfer", {})
        ok = bool(yt) and all(v["model"]["r2"] > v["crop_mean"]["r2"] for v in yt.values())
        c.check("year transfer: model beats crop mean both ways", ok,
                str({k: v["model"]["r2"] for k, v in yt.items()}))
        cov = r.get("quantile_coverage_lusatia")
        c.check("calibrated 10-90% range coverage 0.70..0.92 on Lusatia", cov is not None and 0.70 <= cov <= 0.92, str(cov))
        top = [s_.lower() for s_ in r.get("shap_top5", [])]
        c.check("a water feature in SHAP top 5",
                any(k in t for t in top for k in ("precip", "sm_", "pet")), str(top))
    d = read_csv(DATA_PROCESSED / "demo_predictions.csv")
    c.check("demo_predictions.csv: 3 fields x 11 years x crops", d is not None and
            set(d["field_id"]) == FIELD_IDS and d["year"].nunique() == 11, f"{0 if d is None else len(d)} rows")
    c.check("SHAP figure saved", (DATA_PROCESSED / "figures" / "shap_importance.png").exists())
    return c


def step7():
    c = Checker(7)
    p = ROOT / "data" / "crops.json"
    c.check("data/crops.json exists", p.exists())
    if p.exists():
        d = json.loads(p.read_text())
        crops = d.get("crops", [])
        c.check(">= 12 crops", len(crops) >= 12, f"{len(crops)} crops")
        need = ["sow_months", "harvest_months", "kc", "max_root_depth_m", "drought_tolerance", "family",
                "n_fixing", "ph_range", "ml_crop_code", "price_eur_per_unit"]
        bad = [x["id"] for x in crops if any(k not in x for k in need)]
        c.check("every crop has the required attributes", not bad, f"missing in: {bad}" if bad else "")
        c.check("sources documented", bool(d.get("meta", {}).get("sources")))
        c.check(">= 2 nitrogen-fixing crops", sum(x["n_fixing"] for x in crops) >= 2)
    rp = DATA_PROCESSED / "rotations.json"
    c.check("src/rotations.py exists", (ROOT / "src" / "rotations.py").exists())
    c.check("rotations.json exists (run rotations.py)", rp.exists())
    if rp.exists():
        rot = json.loads(rp.read_text())
        c.check("strategies for every field", FIELD_IDS <= set(rot))
        for fid in sorted(FIELD_IDS & set(rot)):
            st = rot[fid]["strategies"]
            c.check(f"{fid}: current + >= 2 alternatives",
                    len(st) >= 3 and st[0].get("is_current") is True, f"{len(st)} strategies")
            keys = ["score", "why", "evidence", "benefits", "gaps", "risks", "check_locally", "seasons"]
            ok = all(all(k in x for k in keys) and 0 <= x["score"] <= 100 for x in st)
            c.check(f"{fid}: every strategy complete, score 0..100", ok)
            kinds = [x.get("kind") for x in st]
            small = [x for x in st if x.get("kind") == "small_step"]
            c.check(f"{fid}: has a small step (<= 2 crops changed) and a bigger shift",
                    "small_step" in kinds and "bigger_shift" in kinds and all(x["crops_changed"] <= 2 for x in small),
                    str(kinds))
    return c


def step8():
    """Runs the app headless (Streamlit AppTest): all fields load, sliders change the ranking, no exceptions."""
    c = Checker(8)
    app = ROOT / "app" / "app.py"
    c.check("app/app.py exists", app.exists())
    if not app.exists():
        return c
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        c.check("streamlit testing available", False, str(e))
        return c
    at = AppTest.from_file(str(app), default_timeout=180).run()
    c.check("app starts without exceptions", not at.exception, str([e.value for e in at.exception])[:200])
    if at.exception:
        return c
    c.check("4 tabs", len(at.tabs) == 4, str(len(at.tabs)))
    scores = lambda: [m.value for m in at.markdown if "/ 100" in m.value]
    before = scores()
    c.check("3 rotation strategies shown", len(before) == 3, str(before))
    at.slider(key="w_water").set_value(5)
    at.slider(key="w_income").set_value(0)
    at.run()
    after = scores()
    c.check("moving sliders changes the scores, no exceptions", before != after and not at.exception,
            f"{before} -> {after}")
    # forage priority: with livestock/biogas ticked and forage at 5, a forage crop should be suggested
    at.slider(key="w_water").set_value(3)
    at.slider(key="w_income").set_value(3)
    at.slider(key="w_forage").set_value(5)
    at.checkbox(key="forage").check()
    at.run()
    years = " ".join(m.value for m in at.markdown if m.value.startswith("Year "))
    c.check("forage priority brings in a forage crop (lucerne / clover-grass / maize)",
            not at.exception and any(n in years for n in ("Lucerne", "Clover", "Maize")))
    at.slider(key="w_forage").set_value(0)
    at.checkbox(key="forage").uncheck()
    for f in sorted(FIELD_IDS):
        at.selectbox(key="fid").set_value(f)
        at.run()
        c.check(f"{f}: loads without exceptions", not at.exception and len(scores()) == 3)
    src = app.read_text()
    c.check("no Earth Engine calls in the app", "import ee" not in src and "ee.Initialize" not in src)
    return c


def step9():
    c = Checker(9)
    p = DATA_PROCESSED / "dwd_check.json"
    c.check("dwd_check.json exists", p.exists())
    if p.exists():
        r = json.loads(p.read_text())
        c.check("monthly temp correlation >= 0.95", r.get("temp_r", 0) >= 0.95, str(r.get("temp_r")))
        c.check("monthly rain correlation >= 0.6", r.get("rain_r", 0) >= 0.6, str(r.get("rain_r")))
        c.check("station within 30 km of a demo field", r.get("distance_km", 999) <= 30, str(r.get("distance_km")))
        c.check(">= 25 years of overlapping days", r.get("n_days", 0) >= 25 * 365, str(r.get("n_days")))
        c.check("temperature bias within +-1.5 C", abs(r.get("temp_bias_c", 9)) <= 1.5, str(r.get("temp_bias_c")))
        c.check("hot-day and heavy-rain comparison present",
                all(k in r for k in ("hot_days_station_yr", "hot_days_power_yr", "heavy_rain_station_yr")))
    c.check("dwd_monthly.csv exists", (DATA_PROCESSED / "dwd_monthly.csv").exists())
    c.check("app shows the station check", "dwd_check.json" in (ROOT / "app" / "app.py").read_text())
    return c


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8, 9: step9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=STEPS)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not a.step and not a.all:
        ap.error("use --step N or --all")
    steps = list(STEPS) if a.all else [a.step]
    results = [STEPS[s]().report() for s in steps]   # run every step, no short-circuit
    ok = all(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()