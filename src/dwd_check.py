"""Step 9: check NASA POWER against a real ground weather station (German Weather Service, DWD).

Downloads daily climate data (KL product) for the DWD station nearest to the demo fields (Cottbus),
compares it with NASA POWER for the closest demo field, and writes:
  data/processed/dwd_check.json    summary numbers (shown in the app)
  data/processed/dwd_monthly.csv   monthly station vs POWER table (chart in the app)

Usage:
  python src/dwd_check.py                      # find the station by name and download from DWD open data
  python src/dwd_check.py --station-name Cottbus
  python src/dwd_check.py --zip a.zip b.zip    # use zip files you downloaded by hand (same DWD format)

Data: Deutscher Wetterdienst, Climate Data Center, open data (CC BY 4.0).
"""
import argparse
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import FIELDS, RAW_POWER, DATA_PROCESSED  # noqa: E402

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/"
STATION_LIST = BASE + "historical/KL_Tageswerte_Beschreibung_Stationen.txt"
RAW_DWD = RAW_POWER.parent / "dwd"
YEARS = (1991, 2025)                 # comparison period (full years)
HOT, HEAVY = 30.0, 20.0              # same thresholds as the app (hot day > 30 C, heavy rain >= 20 mm)


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def get(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r


def parse_station_list(text):
    """Fixed-width DWD station list -> DataFrame (id, from, to, height, lat, lon, name)."""
    rows = []
    for line in text.splitlines():
        t = line.split()
        if len(t) < 7 or not t[0].isdigit() or len(t[0]) != 5:
            continue                                  # header and dashed separator lines
        try:
            lat, lon = float(t[4]), float(t[5])
        except ValueError:
            continue
        rest = t[6:]
        if rest and rest[-1] in ("Frei", "Free"):     # newer lists end with an 'Abgabe' column
            rest = rest[:-1]
        name = " ".join(rest[:-1]) if len(rest) > 1 else " ".join(rest)   # last token = federal state
        rows.append({"id": t[0], "from": t[1], "to": t[2], "height": float(t[3]),
                     "lat": lat, "lon": lon, "name": name})
    return pd.DataFrame(rows)


def find_station(name):
    print(f"... reading DWD station list ({STATION_LIST})")
    st = parse_station_list(get(STATION_LIST).content.decode("latin-1"))
    lat0 = np.mean([f["lat"] for f in FIELDS])
    lon0 = np.mean([f["lon"] for f in FIELDS])
    st["dist_km"] = [km(lat0, lon0, a, b) for a, b in zip(st["lat"], st["lon"])]
    st["years"] = (st["to"].str[:4].astype(int) - st["from"].str[:4].astype(int))
    st = st[st["to"].str[:4].astype(int) >= YEARS[1]]               # still reporting
    hit = st[st["name"].str.contains(name, case=False)]
    if hit.empty:
        print(f"   no active station named '{name}', using the nearest active station")
        hit = st
    hit = hit.sort_values(["dist_km", "years"], ascending=[True, False])
    s = hit.iloc[0].to_dict()
    print(f"   station {s['id']} {s['name']}  ({s['lat']:.4f}, {s['lon']:.4f}, {s['height']:.0f} m), "
          f"{s['from']}-{s['to']}, {s['dist_km']:.1f} km from the demo fields' centre")
    return s


def download_zips(sid):
    RAW_DWD.mkdir(parents=True, exist_ok=True)
    out = []
    html = get(BASE + "historical/").text
    names = sorted(set(re.findall(rf"tageswerte_KL_{sid}_\d{{8}}_\d{{8}}_hist\.zip", html)))
    if not names:
        raise RuntimeError(f"no historical zip for station {sid} in {BASE}historical/")
    urls = [BASE + "historical/" + names[-1], BASE + f"recent/tageswerte_KL_{sid}_akt.zip"]
    for u in urls:
        p = RAW_DWD / u.rsplit("/", 1)[1]
        try:
            print(f"... {u}")
            p.write_bytes(get(u).content)
            out.append(p)
        except Exception as e:                        # 'recent' is optional
            print(f"   skipped ({e})")
    return out


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.startswith("produkt_klima_tag"))
        df = pd.read_csv(io.BytesIO(z.read(name)), sep=";", encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"MESS_DATUM": "date", "TMK": "t_mean", "TXK": "t_max", "TNK": "t_min",
                            "RSK": "rain"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    df = df[["date", "t_mean", "t_max", "t_min", "rain"]]
    for c in ["t_mean", "t_max", "t_min", "rain"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace(-999.0, np.nan)


def r_(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")


def compare(dwd, station):
    power = pd.read_csv(RAW_POWER / "power_daily.csv", parse_dates=["date"])
    dist = {f["id"]: km(station["lat"], station["lon"], f["lat"], f["lon"]) for f in FIELDS}
    fid = min(dist, key=dist.get)
    p = power[power["field_id"] == fid].rename(columns={
        "T2M": "p_t_mean", "T2M_MAX": "p_t_max", "T2M_MIN": "p_t_min", "PRECTOTCORR": "p_rain"})
    d = dwd.merge(p[["date", "p_t_mean", "p_t_max", "p_t_min", "p_rain"]], on="date", how="inner")
    d = d[(d["date"].dt.year >= YEARS[0]) & (d["date"].dt.year <= YEARS[1])]
    if len(d) < 3650:
        raise RuntimeError(f"only {len(d)} overlapping days: check the station files")

    # monthly: only months with >= 25 valid days on both sides
    d["month"] = d["date"].dt.to_period("M")
    g = d.groupby("month")
    mon = pd.DataFrame({
        "n_t": g.apply(lambda x: (x["t_mean"].notna() & x["p_t_mean"].notna()).sum()),
        "n_r": g.apply(lambda x: (x["rain"].notna() & x["p_rain"].notna()).sum()),
        "t_station": g["t_mean"].mean(), "t_power": g["p_t_mean"].mean(),
        "rain_station": g["rain"].sum(min_count=25), "rain_power": g["p_rain"].sum(min_count=25),
    })
    mon.loc[mon["n_t"] < 25, ["t_station", "t_power"]] = np.nan
    mon.loc[mon["n_r"] < 25, ["rain_station", "rain_power"]] = np.nan
    mon = mon.drop(columns=["n_t", "n_r"]).reset_index()
    mon["month"] = mon["month"].astype(str)

    # yearly counts that the app shows (hot days, heavy rain days, annual rain)
    d["year"] = d["date"].dt.year
    yr = d.groupby("year").agg(
        hot_station=("t_max", lambda s: int((s > HOT).sum())),
        hot_power=("p_t_max", lambda s: int((s > HOT).sum())),
        heavy_station=("rain", lambda s: int((s >= HEAVY).sum())),
        heavy_power=("p_rain", lambda s: int((s >= HEAVY).sum())),
        rain_station=("rain", lambda s: float(s.sum(min_count=330))),
        rain_power=("p_rain", lambda s: float(s.sum(min_count=330))),
    )
    summer = d[d["date"].dt.month.isin([6, 7, 8])].groupby("year").agg(
        s=("rain", lambda s: s.sum(min_count=85)), p=("p_rain", lambda s: s.sum(min_count=85)))

    rs, rp = yr["rain_station"].mean(), yr["rain_power"].mean()
    res = {
        "station_id": station["id"], "station_name": station["name"],
        "station_lat": station["lat"], "station_lon": station["lon"], "station_height_m": station["height"],
        "compared_field": fid, "distance_km": round(dist[fid], 1),
        "period": f"{YEARS[0]}-{YEARS[1]}", "n_days": int(len(d)),
        "temp_r_daily": round(r_(d["t_mean"], d["p_t_mean"]), 3),
        "temp_r": round(r_(mon["t_station"], mon["t_power"]), 3),
        "temp_bias_c": round(float((mon["t_power"] - mon["t_station"]).mean()), 2),
        "tmax_bias_c": round(float((d["p_t_max"] - d["t_max"]).mean()), 2),
        "rain_r_daily": round(r_(d["rain"], d["p_rain"]), 3),
        "rain_r": round(r_(mon["rain_station"], mon["rain_power"]), 3),
        "rain_r_summer": round(r_(summer["s"], summer["p"]), 3),
        "rain_station_mm_yr": round(float(rs), 0), "rain_power_mm_yr": round(float(rp), 0),
        "rain_bias_pct": round(float(100 * (rp - rs) / rs), 1),
        "hot_days_station_yr": round(float(yr["hot_station"].mean()), 1),
        "hot_days_power_yr": round(float(yr["hot_power"].mean()), 1),
        "hot_days_r": round(r_(yr["hot_station"].astype(float), yr["hot_power"].astype(float)), 3),
        "heavy_rain_station_yr": round(float(yr["heavy_station"].mean()), 1),
        "heavy_rain_power_yr": round(float(yr["heavy_power"].mean()), 1),
        "source": "Deutscher Wetterdienst (DWD) Climate Data Center, daily KL, open data CC BY 4.0",
    }
    return res, mon


def verdict(r):
    lines = []
    lines.append(f"Temperature: monthly r = {r['temp_r']}, POWER {r['temp_bias_c']:+.1f} C vs station "
                 f"(daily max {r['tmax_bias_c']:+.1f} C).")
    lines.append(f"Rain: monthly r = {r['rain_r']}, summer r = {r['rain_r_summer']}; POWER {r['rain_power_mm_yr']:.0f} "
                 f"vs station {r['rain_station_mm_yr']:.0f} mm/yr ({r['rain_bias_pct']:+.0f} %).")
    lines.append(f"Hot days > 30 C: POWER {r['hot_days_power_yr']} vs station {r['hot_days_station_yr']} per year "
                 f"(year-to-year r = {r['hot_days_r']}).")
    lines.append(f"Heavy rain days >= 20 mm: POWER {r['heavy_rain_power_yr']} vs station "
                 f"{r['heavy_rain_station_yr']} per year (a ~50 km grid smooths local downpours).")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-name", default="Cottbus")
    ap.add_argument("--zip", nargs="+", help="DWD daily KL zip files downloaded by hand")
    ap.add_argument("--station-id", help="with --zip: 5-digit station id (default: from the zip)")
    ap.add_argument("--lat", type=float, help="with --zip: station latitude")
    ap.add_argument("--lon", type=float, help="with --zip: station longitude")
    a = ap.parse_args()

    if a.zip:
        if a.lat is None or a.lon is None:
            ap.error("--zip needs --lat and --lon of the station (see the station list on DWD open data)")
        sid = a.station_id or re.search(r"KL_(\d{5})_", Path(a.zip[0]).name).group(1)
        station = {"id": sid, "name": a.station_name, "lat": a.lat, "lon": a.lon, "height": float("nan")}
        zips = [Path(z) for z in a.zip]
    else:
        station = find_station(a.station_name)
        zips = download_zips(station["id"])

    dwd = pd.concat([read_zip(z) for z in zips]).drop_duplicates("date", keep="first").sort_values("date")
    print(f"   station days: {len(dwd)} ({dwd['date'].min().date()} to {dwd['date'].max().date()})")

    res, mon = compare(dwd, station)
    res["summary"] = verdict(res)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "dwd_check.json").write_text(json.dumps(res, indent=2))
    mon.to_csv(DATA_PROCESSED / "dwd_monthly.csv", index=False)

    print(f"\n=== NASA POWER ({res['compared_field']}) vs DWD {res['station_name']} "
          f"({res['distance_km']} km apart), {res['period']}, {res['n_days']} days")
    for line in res["summary"]:
        print("  " + line)
    print(f"\nSaved {DATA_PROCESSED / 'dwd_check.json'} and dwd_monthly.csv")


if __name__ == "__main__":
    main()