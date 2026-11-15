"""
Step 2: daily weather for the demo fields from NASA POWER (1981 -> now).
No login needed. One request per field.

Run from the FARMS folder:
    python src/extract_power.py

Output: data/raw/power/power_daily.csv  (one row per field per day)
"""
import time

import pandas as pd
import requests

from config import FIELDS, RAW_POWER, END_DATE

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Agroclimatology parameters (community=AG)
PARAMS = [
    "T2M",                # mean air temp 2 m (C)
    "T2M_MAX",            # daily max (C)  -> heat days
    "T2M_MIN",            # daily min (C)  -> frost days
    "PRECTOTCORR",        # precipitation (mm/day, bias-corrected)
    "RH2M",               # relative humidity (%)
    "WS2M",               # wind speed 2 m (m/s)
    "ALLSKY_SFC_SW_DWN",  # solar radiation (MJ/m2/day with community=AG)
    "GWETROOT",           # root-zone soil wetness (0-1)
    "GWETTOP",            # surface soil wetness (0-1)
]

START = "19810101"
END = END_DATE.replace("-", "")


def fetch(field, retries=3):
    q = {
        "parameters": ",".join(PARAMS),
        "community": "AG",
        "latitude": field["lat"],
        "longitude": field["lon"],
        "start": START,
        "end": END,
        "format": "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=q, timeout=180)
            r.raise_for_status()
            data = r.json()["properties"]["parameter"]
            df = pd.DataFrame(data)              # index = YYYYMMDD, columns = params
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df = df.astype("float").replace(-999.0, float("nan"))
            df.index.name = "date"
            df = df.reset_index()
            df.insert(0, "field_id", field["id"])
            return df
        except Exception as e:
            print(f"   attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"POWER request failed for {field['id']}")


def main():
    frames = []
    for f in FIELDS:
        print(f"... {f['id']}")
        df = fetch(f)
        frames.append(df)
        last_ok = df.dropna(subset=["T2M"])["date"].max().date()
        print(f"  OK  {len(df)} days, data until {last_ok}")
        time.sleep(2)                            # be polite to the API

    out = pd.concat(frames, ignore_index=True)
    path = RAW_POWER / "power_daily.csv"
    out.to_csv(path, index=False)

    print(f"\nSaved {len(out)} rows -> {path}")
    print("\nMissing values per column (%):")
    print((out.isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()