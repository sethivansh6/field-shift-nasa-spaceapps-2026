"""
Step 3b: clean the ECOSTRESS point data from AppEEARS.

Run from the FARMS folder:
    python src/load_ecostress.py

Input : data/raw/ecostress/*-ECO-L2T-LSTE-002-results.csv (+ JET, SM, ESI)
Output: data/processed/ecostress_clean.csv      one row per field per overpass, bad values set to empty
        data/processed/ecostress_qc_report.json how many readings were removed per field and why

Rules
- Cloud, water and quality flags only exist in the LSTE file. ESI, ET and SM come from the same overpass,
  so they are matched to the LSTE flags by field + timestamp. No matching LSTE row = cloud unknown = removed.
- LST kept if: not cloudy, not water, pixel produced (mandatory QA 00/01), good L1B data,
  LST accuracy not "Poor (>2 K)". Converted Kelvin -> C.
- ESI/ET kept if: not cloudy, PET > 0 (PET = 0 means no data, not zero stress), ESI within 0..1.5.
- SM kept if: not cloudy, not water, 0 < SM < 0.6.
- 15 May to 1 July 2025 is a known noisy period (ECOSTRESS known issues): values removed and counted.
- Overpass time varies; solar hour is computed and midday readings (10-15 h solar) are flagged.
"""
import json

import numpy as np
import pandas as pd

from config import RAW_ECOSTRESS, DATA_PROCESSED, FIELDS

NOISY = (pd.Timestamp("2025-05-15", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"))
PRE = {"LSTE": "ECO_L2T_LSTE_002_", "JET": "ECO_L3T_JET_002_", "SM": "ECO_L3T_SM_002_",
       "ESI": "ECO_L4T_ESI_002_"}


def read(product):
    files = list(RAW_ECOSTRESS.glob(f"*-ECO-*{product}-002-results.csv"))
    if not files:
        raise FileNotFoundError(f"No AppEEARS CSV for {product} in {RAW_ECOSTRESS}")
    df = pd.read_csv(files[0], dtype=str)
    df["field_id"] = df["ID"]
    df["time_utc"] = pd.to_datetime(df["Date"].str.replace(" UTC", "", regex=False), utc=True)
    df = df.rename(columns={c: c.replace(PRE[product], "") for c in df.columns})
    return df


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    # ---------------------------------------------------------------- LST + flags
    l = read("LSTE")
    lst = pd.DataFrame({
        "field_id": l["field_id"], "time_utc": l["time_utc"],
        "lst_k": num(l["LST"]),
        "cloud": num(l["cloud"]), "water": num(l["water"]),
        "qa_mandatory": l["QC_Mandatory_QA_flags"],
        "qa_data": l["QC_Data_quality_flag"],
        "qa_lst_acc": l["QC_LST_accuracy"],
    })
    # same point can appear in two overlapping tiles -> keep the row with an LST value first
    lst = (lst.sort_values("lst_k", na_position="last")
              .drop_duplicates(["field_id", "time_utc"], keep="first"))

    # ---------------------------------------------------------------- other products
    def product(name, cols):
        d = read(name)
        out = d[["field_id", "time_utc"]].copy()
        for src, dst in cols.items():
            out[dst] = num(d[src])
        return out.groupby(["field_id", "time_utc"], as_index=False).first()

    esi = product("ESI", {"ESI": "esi", "PET": "pet"})
    jet = product("JET", {"ETdaily": "et_daily"})
    sm = product("SM", {"SM": "sm", "water": "sm_water"})

    df = lst
    for d in (esi, jet, sm):
        df = df.merge(d, on=["field_id", "time_utc"], how="outer")
    df = df.sort_values(["field_id", "time_utc"]).reset_index(drop=True)

    # ---------------------------------------------------------------- flags
    no_lste_row = df["cloud"].isna()
    cloudy = (df["cloud"] == 1) | no_lste_row
    water = df["water"] == 1
    produced = df["qa_mandatory"].isin(["0b00", "0b01"])
    good_l1b = df["qa_data"] == "0b00"
    poor_acc = df["qa_lst_acc"] == "0b00"
    noisy = df["time_utc"].between(*NOISY)

    lst_ok = df["lst_k"].notna() & ~cloudy & ~water & produced & good_l1b & ~poor_acc & ~noisy
    esi_ok = df["esi"].notna() & ~cloudy & (df["pet"] > 0) & df["esi"].between(0, 1.5) & ~noisy
    et_ok = df["et_daily"].notna() & ~cloudy & (df["pet"].fillna(1) > 0) & (df["et_daily"] >= 0) & ~noisy
    sm_ok = df["sm"].notna() & ~cloudy & (df["sm_water"].fillna(0) == 0) & df["sm"].between(0.001, 0.6) & ~noisy

    # ---------------------------------------------------------------- solar time
    lon = df["field_id"].map({f["id"]: f["lon"] for f in FIELDS})
    solar = df["time_utc"] + pd.to_timedelta(lon / 15, unit="h")
    df["solar_hour"] = (solar.dt.hour + solar.dt.minute / 60).round(2)
    df["midday"] = df["solar_hour"].between(10, 15)

    clean = pd.DataFrame({
        "field_id": df["field_id"],
        "time_utc": df["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": df["time_utc"].dt.strftime("%Y-%m-%d"),
        "solar_hour": df["solar_hour"],
        "midday": df["midday"],
        "lst_c": np.where(lst_ok, df["lst_k"] - 273.15, np.nan),
        "esi": np.where(esi_ok, df["esi"], np.nan),
        "pet": np.where(esi_ok, df["pet"], np.nan),
        "et_daily": np.where(et_ok, df["et_daily"], np.nan),
        "sm": np.where(sm_ok, df["sm"], np.nan),
    })
    clean = clean.dropna(subset=["lst_c", "esi", "et_daily", "sm"], how="all")
    out = DATA_PROCESSED / "ecostress_clean.csv"
    clean.to_csv(out, index=False)

    # ---------------------------------------------------------------- QC report
    report = {}
    for fid, g in df.groupby("field_id"):
        idx = g.index
        report[fid] = {
            "overpasses_total": int(len(g)),
            "removed_cloudy": int(cloudy[idx].sum()),
            "removed_no_flag_row": int(no_lste_row[idx].sum()),
            "removed_water": int(water[idx].sum()),
            "removed_noisy_period_2025": int(noisy[idx].sum()),
            "lst_poor_accuracy_or_bad_qa": int((df.loc[idx, "lst_k"].notna() & ~cloudy[idx]
                                                & ~(produced & good_l1b & ~poor_acc)[idx]).sum()),
            "kept_lst": int(lst_ok[idx].sum()),
            "kept_esi": int(esi_ok[idx].sum()),
            "kept_et_daily": int(et_ok[idx].sum()),
            "kept_sm": int(sm_ok[idx].sum()),
            "kept_esi_midday": int((esi_ok & df["midday"])[idx].sum()),
            "share_cloudy_pct": round(100 * cloudy[idx].mean(), 1),
            "first": g["time_utc"].min().strftime("%Y-%m-%d"),
            "last": g["time_utc"].max().strftime("%Y-%m-%d"),
        }
    (DATA_PROCESSED / "ecostress_qc_report.json").write_text(json.dumps(report, indent=2))

    # ---------------------------------------------------------------- summary
    print(f"Saved {len(clean)} rows -> {out}\n")
    print(pd.DataFrame(report).T[["overpasses_total", "share_cloudy_pct", "kept_lst", "kept_esi",
                                  "kept_esi_midday", "kept_et_daily", "kept_sm"]].to_string())
    summer = clean[pd.to_datetime(clean["date"]).dt.month.isin([6, 7, 8])].copy()
    summer["year"] = summer["date"].str[:4]
    print("\nMean summer ESI by year (lower = more water stress):")
    print(summer.pivot_table(index="year", columns="field_id", values="esi", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()