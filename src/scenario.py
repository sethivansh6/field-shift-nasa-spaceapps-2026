"""
Climate scenarios ("what if summers are drier / hotter?") for the app.

A scenario changes each past season's weather (2015-2025) and the field's monthly climate, then the
same ML model and rotation scorer are re-run. Simple, transparent "delta" method:

  summer rain change dR (e.g. -0.2 = 20 % less rain in June-August)
    precip_summer         x (1 + dR)
    root-zone moisture    x (1 + 0.4 dR)  (soil moisture reacts less than rain; rough elasticity)
    monthly rain Jun-Aug  x (1 + dR)      (for crop water deficits)
  warming dT (C)
    heat days > 30 C      recounted from NASA POWER daily max temperature + dT, for each field and season
    late frost days       recounted from NASA POWER daily min temperature + dT (15 Apr - 31 May)
    growing degree days   + dT x 153 days (March-July)
    PET / ET0             x (1 + 0.04 dT) (reference evapotranspiration rises ~3-5 % per C)

These are first-order approximations for exploring sensitivity, not climate projections.
"""
import numpy as np
import pandas as pd

from config import RAW_POWER


def load_power():
    p = pd.read_csv(RAW_POWER / "power_daily.csv", parse_dates=["date"])
    return p


def _count(power, fid, year, col, months, thr, above, dT, day_from=None):
    d = power[(power["field_id"] == fid) & (power["date"].dt.year == year) & power["date"].dt.month.isin(months)]
    if day_from is not None:
        d = d[(d["date"].dt.month > day_from[0]) | (d["date"].dt.day >= day_from[1])]
    v = d[col] + dT
    return int((v > thr).sum()) if above else int((v < thr).sum())


def perturb_seasons(seasons, power, dR=0.0, dT=0.0):
    s = seasons.copy()
    for col in ("precip_summer", "sm_rz_mj", "sm_rz_anom_mj", "gdd", "pet_summer", "heat_days", "late_frost_days"):
        s[col] = s[col].astype(float)
    if dR:
        s["precip_summer"] = s["precip_summer"] * (1 + dR)
        add = s["sm_rz_mj"] * 0.4 * dR
        s["sm_rz_mj"] = s["sm_rz_mj"] + add
        s["sm_rz_anom_mj"] = s["sm_rz_anom_mj"] + add
    if dT:
        s["gdd"] = s["gdd"] + dT * 153
        s["pet_summer"] = s["pet_summer"] * (1 + 0.04 * dT)
        for i, r in s.iterrows():
            if r["field_id"] not in set(power["field_id"]):
                continue
            base = _count(power, r["field_id"], r["year"], "T2M_MAX", [6, 7, 8], 30, True, 0)
            new = _count(power, r["field_id"], r["year"], "T2M_MAX", [6, 7, 8], 30, True, dT)
            s.at[i, "heat_days"] = r["heat_days"] * new / base if base > 0 else r["heat_days"] + new
            fb = _count(power, r["field_id"], r["year"], "T2M_MIN", [4, 5], 0, False, 0, (4, 15))
            fn = _count(power, r["field_id"], r["year"], "T2M_MIN", [4, 5], 0, False, dT, (4, 15))
            s.at[i, "late_frost_days"] = r["late_frost_days"] * fn / fb if fb > 0 else r["late_frost_days"]
    return s


def perturb_monthly(monthly, dR=0.0, dT=0.0):
    m = monthly.copy()
    if dR:
        summer = m["month"].isin([6, 7, 8])
        m.loc[summer, "rain_mm"] = m.loc[summer, "rain_mm"] * (1 + dR)
    if dT:
        m["et0_mm"] = m["et0_mm"] * (1 + 0.04 * dT)
    m["water_balance_mm"] = m["rain_mm"] - m["et0_mm"]
    return m