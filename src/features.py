"""
Step 5: field profiles for the app (the 3 demo fields), from steps 1-3 data.

Run from the FARMS folder:
    python src/features.py

Outputs (data/processed/):
    field_profiles.json      one summary per field: what the app screens show
    monthly_field_table.csv  field x month (1-12) normals, for charts
    yearly_field_table.csv   field x year, for trend charts and the past-season story

Normal period = 1991-2020 (WMO standard). Recent = 2011-2025. Trends use linear regression
and are reported per decade with a p-value, so only real changes are called "trends".
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import FIELDS, RAW_GEE, RAW_POWER, DATA_PROCESSED

NORMAL = (1991, 2020)
RECENT = (2011, 2025)
LAST_FULL_YEAR = 2025
FIELD = {f["id"]: f for f in FIELDS}


# ================================================================ helpers
def trend(years, values):
    """Slope per decade, p-value, n. None if too few points."""
    s = pd.Series(values, index=years).dropna()
    if len(s) < 5:
        return None
    r = stats.linregress(s.index.astype(float), s.values)
    return {"per_decade": round(float(r.slope * 10), 3), "p_value": round(float(r.pvalue), 4),
            "significant": bool(r.pvalue < 0.05), "years": [int(s.index.min()), int(s.index.max())]}


def doy_to_date(doy):
    if doy is None or np.isnan(doy):
        return None
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(round(doy)) - 1)).strftime("%d %b")


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def saxton_rawls_awc(sand, clay, soc):
    """Same formula as extract_training.py (Saxton & Rawls 2006), % volume."""
    S, C, OM = sand / 100, clay / 100, soc * 1.724
    t1500 = -0.024*S + 0.487*C + 0.006*OM + 0.005*S*OM - 0.013*C*OM + 0.068*S*C + 0.031
    th1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251*S + 0.195*C + 0.011*OM + 0.006*S*OM - 0.027*C*OM + 0.452*S*C + 0.299
    th33 = t33 + (1.283 * t33**2 - 0.374 * t33 - 0.015)
    return (th33 - th1500) * 100


# ================================================================ FAO-56 reference ET0
def et0_fao56(d, lat, elev):
    """Daily Penman-Monteith reference evapotranspiration (mm/day), FAO-56 eqs. 7-39."""
    tmax, tmin = d["T2M_MAX"], d["T2M_MIN"]
    t = (tmax + tmin) / 2
    u2 = d["WS2M"]
    rs = d["ALLSKY_SFC_SW_DWN"]                       # MJ/m2/day (community=AG)
    e0 = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    es = (e0(tmax) + e0(tmin)) / 2
    ea = d["RH2M"] / 100 * es
    delta = 4098 * e0(t) / (t + 237.3) ** 2
    p = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * p
    j = d["date"].dt.day_of_year
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dec = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra
    rns = 0.77 * rs
    ratio = np.clip(rs / rso, 0.3, 1.0)
    rnl = 4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (t + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return et0.clip(lower=0)


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ================================================================ POWER (daily, 1981+)
def power_features(p, fid, elev):
    d = p[p["field_id"] == fid].copy()
    d["year"], d["month"], d["doy"] = d["date"].dt.year, d["date"].dt.month, d["date"].dt.day_of_year
    d["et0"] = et0_fao56(d, FIELD[fid]["lat"], elev)
    d = d[d["year"] <= LAST_FULL_YEAR]

    yearly = []
    for y, g in d.groupby("year"):
        spring = g[(g["month"] <= 6) & (g["T2M_MIN"] < 0)]
        autumn = g[(g["month"] >= 7) & (g["T2M_MIN"] < 0)]
        season = g[g["month"].between(4, 8)]
        vegp = g[g["month"].between(4, 9)]
        yearly.append({
            "year": y,
            "last_spring_frost_doy": spring["doy"].max() if len(spring) else np.nan,
            "first_autumn_frost_doy": autumn["doy"].min() if len(autumn) else np.nan,
            "heat_days_30": int((g["T2M_MAX"] > 30).sum()),
            "warm_nights_20": int((g["T2M_MIN"] > 20).sum()),
            "t_mean": g["T2M"].mean(),
            "rain_year_mm": g["PRECTOTCORR"].sum(),
            "rain_apr_aug_mm": season["PRECTOTCORR"].sum(),
            "longest_dry_spell_apr_aug": longest_run(season["PRECTOTCORR"] < 1),
            "heavy_rain_days_20mm": int((g["PRECTOTCORR"] > 20).sum()),
            "et0_year_mm": g["et0"].sum(min_count=300),
            "water_balance_apr_sep_mm": vegp["PRECTOTCORR"].sum() - vegp["et0"].sum(min_count=150),
            "gwetroot_apr_sep": vegp["GWETROOT"].mean(),
        })
    yr = pd.DataFrame(yearly).set_index("year")
    yr["frost_free_days"] = yr["first_autumn_frost_doy"] - yr["last_spring_frost_doy"]

    norm = yr.loc[NORMAL[0]:NORMAL[1]]
    rec = yr.loc[RECENT[0]:RECENT[1]]
    dn = d[d["year"].between(*NORMAL)]
    nyears = NORMAL[1] - NORMAL[0] + 1
    monthly = pd.DataFrame({
        "rain_mm": dn.groupby("month")["PRECTOTCORR"].sum() / nyears,
        "et0_mm": dn.groupby("month")["et0"].sum() / nyears,
        "tmax_c": dn.groupby("month")["T2M_MAX"].mean(),
        "tmin_c": dn.groupby("month")["T2M_MIN"].mean(),
        "frost_days": dn.groupby("month")["T2M_MIN"].apply(lambda s: (s < 0).sum()) / nyears,
        "heat_days_30_normal": dn.groupby("month")["T2M_MAX"].apply(lambda s: (s > 30).sum()) / nyears,
    })
    dr_ = d[d["year"].between(*RECENT)]
    monthly["heat_days_30_recent"] = dr_.groupby("month")["T2M_MAX"].apply(lambda s: (s > 30).sum()) / (RECENT[1] - RECENT[0] + 1)
    monthly["water_balance_mm"] = monthly["rain_mm"] - monthly["et0_mm"]

    ys = yr.index
    prof = {
        "frost": {
            "last_spring_median": doy_to_date(norm["last_spring_frost_doy"].median()),
            "last_spring_1_in_10": doy_to_date(norm["last_spring_frost_doy"].quantile(0.9)),
            "first_autumn_median": doy_to_date(norm["first_autumn_frost_doy"].median()),
            "first_autumn_1_in_10": doy_to_date(norm["first_autumn_frost_doy"].quantile(0.1)),
            "last_spring_median_doy": r(norm["last_spring_frost_doy"].median(), 0),
            "frost_free_days_normal": r(norm["frost_free_days"].mean(), 0),
            "frost_free_days_recent": r(rec["frost_free_days"].mean(), 0),
            "frost_free_trend": trend(ys, yr["frost_free_days"]),
        },
        "heat": {
            "heat_days_30_normal": r(norm["heat_days_30"].mean()),
            "heat_days_30_recent": r(rec["heat_days_30"].mean()),
            "heat_days_trend": trend(ys, yr["heat_days_30"]),
            "warm_nights_recent": r(rec["warm_nights_20"].mean()),
            "t_mean_normal": r(norm["t_mean"].mean()),
            "t_mean_trend": trend(ys, yr["t_mean"]),
        },
        "water": {
            "rain_year_normal_mm": r(norm["rain_year_mm"].mean(), 0),
            "rain_apr_aug_normal_mm": r(norm["rain_apr_aug_mm"].mean(), 0),
            "rain_apr_aug_cv_pct": r(100 * norm["rain_apr_aug_mm"].std() / norm["rain_apr_aug_mm"].mean(), 0),
            "rain_apr_aug_trend": trend(ys, yr["rain_apr_aug_mm"]),
            "dry_spell_median_days": r(norm["longest_dry_spell_apr_aug"].median(), 0),
            "dry_spell_recent_median_days": r(rec["longest_dry_spell_apr_aug"].median(), 0),
            "dry_spell_trend": trend(ys, yr["longest_dry_spell_apr_aug"]),
            "heavy_rain_days_recent": r(rec["heavy_rain_days_20mm"].mean()),
            "et0_year_normal_mm": r(norm["et0_year_mm"].mean(), 0),
            "et0_trend": trend(ys, yr["et0_year_mm"]),
            "water_balance_apr_sep_normal_mm": r(norm["water_balance_apr_sep_mm"].mean(), 0),
            "water_balance_apr_sep_recent_mm": r(rec["water_balance_apr_sep_mm"].mean(), 0),
            "water_balance_trend": trend(ys, yr["water_balance_apr_sep_mm"]),
            "power_soil_wetness_trend": trend(ys, yr["gwetroot_apr_sep"]),
            "note": "NASA POWER grid is ~50 x 60 km; field_south and field_southeast share one cell.",
        },
    }
    return prof, monthly, yr


# ================================================================ satellite datasets
def gee(name):
    df = pd.read_csv(RAW_GEE / f"{name}.csv")
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    return df


def smap_features(fid):
    s = gee("smap_monthly")
    s = s[s["field_id"] == fid].copy()
    s["year"], s["month"] = s["date"].dt.year, s["date"].dt.month
    monthly = s.groupby("month")["sm_rootzone"].mean()
    veg = s[s["month"].between(4, 9)].groupby("year")["sm_rootzone"].mean()
    veg = veg[veg.index <= LAST_FULL_YEAR]
    return {
        "sm_rootzone_apr_sep_mean": r(veg.mean(), 3),
        "driest_year": int(veg.idxmin()), "wettest_year": int(veg.idxmax()),
        "trend": trend(veg.index, veg.values),
        "years": [int(veg.index.min()), int(veg.index.max())],
    }, monthly, veg


def imerg_features(fid):
    i = gee("imerg_monthly")
    i = i[i["field_id"] == fid].copy()
    i["year"], i["month"] = i["date"].dt.year, i["date"].dt.month
    aa = i[i["month"].between(4, 8)].groupby("year")["precip_mm"].sum()
    aa = aa[(aa.index >= 2001) & (aa.index <= 2024)]
    return {"rain_apr_aug_mean_mm": r(aa.mean(), 0),
            "rain_apr_aug_cv_pct": r(100 * aa.std() / aa.mean(), 0),
            "years": [int(aa.index.min()), int(aa.index.max())]}, aa


def ndvi_features(fid):
    n = gee("modis_ndvi")
    n = n[(n["field_id"] == fid) & (n["SummaryQA"] <= 1)].copy()
    n["year"], n["month"] = n["date"].dt.year, n["date"].dt.month
    mmax = n.groupby(["year", "month"])["NDVI"].max().reset_index()
    per_year = mmax.groupby("year").agg(
        months_with_data=("NDVI", "size"),
        bare_months=("NDVI", lambda s: int((s < 0.3).sum())),
    )
    per_year["peak_ndvi"] = n[n["month"].between(4, 8)].groupby("year")["NDVI"].max()
    per_year = per_year[(per_year.index >= 2001) & (per_year.index <= LAST_FULL_YEAR)]
    ok = per_year[per_year["months_with_data"] >= 8]
    monthly = n.groupby("month")["NDVI"].median()
    return {
        "bare_months_per_year": r(ok["bare_months"].mean()),
        "bare_months_recent": r(ok.loc[RECENT[0]:]["bare_months"].mean()),
        "bare_months_trend": trend(ok.index, ok["bare_months"]),
        "peak_ndvi_mean": r(per_year["peak_ndvi"].mean(), 2),
        "peak_ndvi_by_year": {int(y): r(v, 2) for y, v in per_year["peak_ndvi"].items()},
        "note": "Bare month = monthly max NDVI < 0.3 (good/marginal QA only); years with < 8 months of data skipped.",
    }, monthly, per_year


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def s2_bare_features(fid):
    """Bare-soil months from Sentinel-2 (10 m). Bare month = median NDVI of clear images < 0.3."""
    p = RAW_GEE / "s2_ndvi.csv"
    if not p.exists():
        return None, None, None
    s2 = pd.read_csv(p, parse_dates=["date"])
    s2 = s2[s2["field_id"] == fid].copy()
    s2["year"], s2["month"] = s2["date"].dt.year, s2["date"].dt.month
    m = s2.groupby(["year", "month"]).agg(ndvi=("ndvi_median", "median"), n=("ndvi_median", "size"),
                                          bare_scl=("bare_scl_frac", "mean")).reset_index()
    m["bare"] = m["ndvi"] < 0.3
    per_year = m.groupby("year").agg(months_with_data=("bare", "size"), bare_months=("bare", "sum"))
    per_year = per_year[(per_year.index <= LAST_FULL_YEAR)]
    ok = per_year[per_year["months_with_data"] >= 8]
    # share of years in which each calendar month was bare
    month_share = m[m["year"] <= LAST_FULL_YEAR].groupby("month")["bare"].mean()
    typical = [MONTHS[i - 1] for i, v in month_share.items() if v >= 0.5]
    prof = {
        "bare_months_per_year": r(ok["bare_months"].mean()),
        "bare_months_by_year": {int(y): int(v) for y, v in ok["bare_months"].items()},
        "usually_bare_months": typical,
        "bare_share_by_month": {MONTHS[i - 1]: r(v, 2) for i, v in month_share.items()},
        "years_used": [int(ok.index.min()), int(ok.index.max())] if len(ok) else None,
        "clear_images": int(len(s2)),
        "note": ("Sentinel-2 10 m, clear pixels within 40 m of the field point. Bare month = monthly median "
                 "NDVI < 0.3. Years with < 8 months of clear data are skipped."),
    }
    return prof, month_share, per_year


def et_lst_features(fid):
    et = gee("modis_et")
    et = et[et["field_id"] == fid].copy()
    et["year"], et["month"] = et["date"].dt.year, et["date"].dt.month
    su = et[et["month"].between(6, 8)].groupby("year")[["ET", "PET"]].sum()
    su["et_pet_ratio"] = su["ET"] / su["PET"]
    su = su[su.index <= LAST_FULL_YEAR]

    lst = gee("modis_lst")
    lst = lst[lst["field_id"] == fid].copy()
    lst["year"], lst["month"] = lst["date"].dt.year, lst["date"].dt.month
    sl = lst[lst["month"].between(6, 8)].groupby("year")["LST_Day_1km"].agg(["mean", lambda s: int((s > 35).sum())])
    sl.columns = ["summer_lst_day_c", "hot_surface_composites_35c"]
    sl = sl[sl.index <= LAST_FULL_YEAR]
    return {
        "summer_et_pet_ratio_mean": r(su["et_pet_ratio"].mean(), 2),
        "summer_et_pet_ratio_by_year": {int(y): r(v, 2) for y, v in su["et_pet_ratio"].items()},
        "summer_lst_day_mean_c": r(sl["summer_lst_day_c"].mean()),
        "summer_lst_trend": trend(sl.index, sl["summer_lst_day_c"]),
        "note": "ET/PET < 1 means the crop used less water than the air demanded (water stress).",
    }, su, sl


def grace_features(fid):
    g = gee("grace_mascon")
    g = g[g["field_id"] == fid].dropna(subset=["lwe_thickness"])
    yrs = g["date"].dt.year + (g["date"].dt.day_of_year - 1) / 365.25
    res = stats.linregress(yrs, g["lwe_thickness"])
    return {"trend_cm_per_year": r(res.slope, 2), "p_value": r(res.pvalue, 4),
            "years": [int(g["date"].dt.year.min()), int(g["date"].dt.year.max())],
            "note": "Regional total water storage (~300 km). Context only, not field level."}


def soil_terrain(fid):
    s = gee("soilgrids").set_index("field_id").loc[fid]

    def avg(prop, div):
        return (s[f"{prop}_0-5cm_mean"] * 5 + s[f"{prop}_5-15cm_mean"] * 10 + s[f"{prop}_15-30cm_mean"] * 15) / 30 / div

    sand, clay, soc = avg("sand", 10), avg("clay", 10), avg("soc", 100)
    silt = avg("silt", 10)
    t = gee("terrain").set_index("field_id").loc[fid]
    awc = saxton_rawls_awc(sand, clay, soc)
    texture = "sand" if sand >= 85 else "loamy sand" if sand >= 70 else "sandy loam" if sand >= 50 and clay < 20 \
        else "clay" if clay >= 40 else "loam"
    return {
        "sand_pct": r(sand), "silt_pct": r(silt), "clay_pct": r(clay), "soc_pct": r(soc, 2),
        "ph": r(avg("phh2o", 10)), "bulk_density_g_cm3": r(avg("bdod", 100), 2),
        "texture_class": texture,
        "awc_pct_volume": r(awc), "awc_mm_top_60cm": r(awc / 100 * 600, 0),
        "elevation_m": r(t["elevation"], 0), "slope_deg": r(t["slope_deg"], 1),
        "source": "SoilGrids 250 m (modelled, 0-30 cm weighted mean); farmer soil test overrides this.",
    }


def crops(fid):
    c = gee("eucropmap")
    c = c[c["field_id"] == fid].sort_values("year")
    return {int(r_.year): (getattr(r_, "crop_name", None) or str(r_.crop_code)) for r_ in c.itertuples()}


def ecostress_features(fid):
    p = DATA_PROCESSED / "ecostress_clean.csv"
    if not p.exists():
        return None, None
    e = pd.read_csv(p)
    e = e[e["field_id"] == fid].copy()
    e["date"] = pd.to_datetime(e["date"])
    e["year"], e["month"] = e["date"].dt.year, e["date"].dt.month
    su = e[e["month"].between(6, 8)]
    by = su.groupby("year").agg(esi_mean=("esi", "mean"), esi_n=("esi", "count"),
                                esi_midday_n=("esi", lambda s: int(s[su.loc[s.index, "midday"]].notna().sum())))
    by["confidence"] = np.where((by.index <= 2019) | (by["esi_n"] < 5), "low", "ok")
    ok = by[by["confidence"] == "ok"]
    prof = {
        "summer_esi_by_year": {int(y): {"esi": r(v.esi_mean, 2), "n": int(v.esi_n), "confidence": v.confidence}
                               for y, v in by.iterrows()},
        "most_stressed_summer": int(ok["esi_mean"].idxmin()) if len(ok) else None,
        "least_stressed_summer": int(ok["esi_mean"].idxmax()) if len(ok) else None,
        "note": ("ESI = actual / potential evapotranspiration (lower = more water stress). 2018-2019 marked "
                 "low confidence: mission start 30 Jul 2018 (summer = August, after harvest) and early "
                 "instrument issues. Summers with < 5 clear readings also low confidence."),
    }
    return prof, by


# ================================================================ main
def main():
    p = pd.read_csv(RAW_POWER / "power_daily.csv", parse_dates=["date"])
    profiles, monthly_rows, yearly_rows = {}, [], []
    qc = json.loads((DATA_PROCESSED / "ecostress_qc_report.json").read_text()) \
        if (DATA_PROCESSED / "ecostress_qc_report.json").exists() else {}

    for f in FIELDS:
        fid = f["id"]
        soil = soil_terrain(fid)
        pw, pm, py = power_features(p, fid, soil["elevation_m"] or 70)
        sm, sm_month, sm_year = smap_features(fid)
        im, im_year = imerg_features(fid)
        nd, nd_month, nd_year = ndvi_features(fid)
        et, et_year, lst_year = et_lst_features(fid)
        eco, eco_year = ecostress_features(fid)
        s2, s2_month, s2_year = s2_bare_features(fid)

        profiles[fid] = {
            "label": f["label"], "lat": f["lat"], "lon": f["lon"],
            "crops_eucropmap": crops(fid),
            **pw,
            "soil_moisture_smap": sm,
            "rain_imerg": im,
            "vegetation_modis": nd,
            "bare_soil_sentinel2": s2,
            "water_use_heat_modis": et,
            "groundwater_grace": grace_features(fid),
            "soil_terrain": soil,
            "stress_ecostress": eco,
            "ecostress_quality": qc.get(fid),
        }

        m = pm.copy()
        m["smap_sm_rootzone"] = sm_month
        m["ndvi_median"] = nd_month
        if s2_month is not None:
            m["s2_bare_share"] = s2_month
        m.insert(0, "month", m.index)
        m.insert(0, "field_id", fid)
        monthly_rows.append(m)

        y = py.copy()
        y = y.join(sm_year.rename("smap_sm_apr_sep"), how="left")
        y = y.join(im_year.rename("imerg_rain_apr_aug_mm"), how="left")
        y = y.join(nd_year[["bare_months", "peak_ndvi", "months_with_data"]], how="left")
        if s2_year is not None:
            y = y.join(s2_year[["bare_months"]].rename(columns={"bare_months": "s2_bare_months"}), how="left")
        y = y.join(et_year[["et_pet_ratio"]].rename(columns={"et_pet_ratio": "modis_et_pet_summer"}), how="left")
        y = y.join(lst_year, how="left")
        if eco_year is not None:
            y = y.join(eco_year.rename(columns={"esi_mean": "ecostress_esi_summer",
                                                "esi_n": "ecostress_n", "confidence": "ecostress_confidence"})
                       [["ecostress_esi_summer", "ecostress_n", "ecostress_confidence"]], how="left")
        y.insert(0, "year", y.index)
        y.insert(0, "field_id", fid)
        yearly_rows.append(y)

    (DATA_PROCESSED / "field_profiles.json").write_text(json.dumps(profiles, indent=2, default=str))
    pd.concat(monthly_rows).to_csv(DATA_PROCESSED / "monthly_field_table.csv", index=False)
    pd.concat(yearly_rows).to_csv(DATA_PROCESSED / "yearly_field_table.csv", index=False)

    # ---------------------------------------------------------------- summary
    print("Saved field_profiles.json, monthly_field_table.csv, yearly_field_table.csv\n")
    rows = []
    for fid, pr in profiles.items():
        rows.append({
            "field": fid,
            "last frost (median / 1-in-10)": f"{pr['frost']['last_spring_median']} / {pr['frost']['last_spring_1_in_10']}",
            "heat days >30 (normal->recent)": f"{pr['heat']['heat_days_30_normal']} -> {pr['heat']['heat_days_30_recent']}",
            "rain Apr-Aug mm": pr["water"]["rain_apr_aug_normal_mm"],
            "ET0 mm/yr": pr["water"]["et0_year_normal_mm"],
            "balance Apr-Sep (normal->recent)": f"{pr['water']['water_balance_apr_sep_normal_mm']} -> {pr['water']['water_balance_apr_sep_recent_mm']}",
            "bare months (S2 / MODIS)": f"{(pr['bare_soil_sentinel2'] or {}).get('bare_months_per_year')} / "
                                        f"{pr['vegetation_modis']['bare_months_per_year']}",
            "usually bare": ",".join((pr["bare_soil_sentinel2"] or {}).get("usually_bare_months") or []) or "-",
            "texture / AWC %": f"{pr['soil_terrain']['texture_class']} / {pr['soil_terrain']['awc_pct_volume']}",
            "GRACE cm/yr": pr["groundwater_grace"]["trend_cm_per_year"],
        })
    print(pd.DataFrame(rows).set_index("field").T.to_string())


if __name__ == "__main__":
    main()