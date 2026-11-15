"""
Step 1: pull satellite data for the demo fields from Google Earth Engine.

Run from the FARMS folder:
    python src/extract_gee.py

Output: one CSV per dataset in data/raw/gee/
Each dataset runs on its own, so if one fails the others still finish.
Values are the pixel that contains each field point (native resolution).
"""
import json
import traceback

import ee
import pandas as pd

from config import GEE_PROJECT, FIELDS, END_DATE, RAW_GEE


# ---------------------------------------------------------------- setup
def init_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        print("Earth Engine init failed.\n"
              "1) run: earthengine authenticate\n"
              "2) check GEE_PROJECT in src/config.py")
        raise
    print(f"Earth Engine ready (project: {GEE_PROJECT})")


def fields_fc(buffer_m=0):
    feats = []
    for f in FIELDS:
        geom = ee.Geometry.Point([f["lon"], f["lat"]])
        if buffer_m:
            geom = geom.buffer(buffer_m)
        feats.append(ee.Feature(geom, {"field_id": f["id"]}))
    return ee.FeatureCollection(feats)


def to_df(fc):
    """Bring a FeatureCollection back as a DataFrame (handles paging)."""
    try:
        df = ee.data.computeFeatures(
            {"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    except Exception:
        info = fc.getInfo()
        df = pd.DataFrame([f["properties"] for f in info["features"]])
    return df.drop(columns=[c for c in ("geo",) if c in df.columns])


# ---------------------------------------------------------------- sampling helpers
def sample_series(ic, bands, scale, fc):
    """Value of each band at each field, for every image in the collection."""
    reducer = ee.Reducer.first()
    if len(bands) == 1:
        reducer = reducer.setOutputs(bands)

    def per_image(img):
        d = img.date().format("YYYY-MM-dd")
        out = img.select(bands).reduceRegions(collection=fc, reducer=reducer, scale=scale)
        return out.map(lambda f: f.set("date", d))

    return to_df(ic.map(per_image).flatten())


def monthly_means(ic, start, end):
    """Turn a high-frequency collection (e.g. SMAP 3-hourly) into monthly mean images."""
    start = ee.Date(start)
    n_months = ee.Date(end).difference(start, "month").floor()

    def make(i):
        s = start.advance(i, "month")
        e = s.advance(1, "month")
        sub = ic.filterDate(s, e)
        return (sub.mean()
                .set("system:time_start", s.millis())
                .set("n_images", sub.size()))

    months = ee.List.sequence(0, n_months.subtract(1))
    return ee.ImageCollection(months.map(make)).filter(ee.Filter.gt("n_images", 0))


# ---------------------------------------------------------------- time series datasets
SERIES = {
    # name: (collection id, bands, scale m, start date, monthly composite?)
    "smap_monthly":   ("NASA/SMAP/SPL4SMGP/008",
                       ["sm_surface", "sm_rootzone", "sm_rootzone_pctl"], 11000, "2015-04-01", True),
    "imerg_monthly":  ("NASA/GPM_L3/IMERG_MONTHLY_V07",
                       ["precipitation"], 11132, "2000-06-01", False),
    "modis_ndvi":     ("MODIS/061/MOD13Q1",
                       ["NDVI", "EVI", "SummaryQA"], 250, "2000-02-18", False),
    "modis_lst":      ("MODIS/061/MOD11A2",
                       ["LST_Day_1km", "LST_Night_1km"], 1000, "2000-02-18", False),
    "modis_et":       ("MODIS/061/MOD16A2GF",
                       ["ET", "PET"], 500, "2000-01-01", False),
    "grace_mascon":   ("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI",
                       ["lwe_thickness"], 55660, "2002-04-01", False),
}

# Convert raw stored integers to real units after download
SCALE_FIX = {
    "modis_ndvi": {"NDVI": 0.0001, "EVI": 0.0001},
    "modis_et":   {"ET": 0.1, "PET": 0.1},          # kg/m2 (= mm) per 8 days
}


def postprocess(name, df):
    for col, factor in SCALE_FIX.get(name, {}).items():
        if col in df:
            df[col] = df[col] * factor
    if name == "modis_lst":                          # 0.02 K -> deg C
        for col in ("LST_Day_1km", "LST_Night_1km"):
            if col in df:
                df[col] = df[col] * 0.02 - 273.15
    if name == "imerg_monthly" and "precipitation" in df:
        # monthly mean rate in mm/hr -> total mm in that month
        d = pd.to_datetime(df["date"])
        df["precip_mm"] = df["precipitation"] * 24 * d.dt.days_in_month
    if "first" in df.columns:
        df = df.rename(columns={"first": "value"})
    return df.sort_values(["field_id", "date"]).reset_index(drop=True)


def run_series(name, cid, bands, scale, start, monthly):
    ic = ee.ImageCollection(cid).filterDate(start, END_DATE).select(bands)
    if monthly:
        ic = monthly_means(ic, start, END_DATE)
    df = sample_series(ic, bands, scale, fields_fc())
    return postprocess(name, df)


# ---------------------------------------------------------------- static datasets
def run_terrain():
    dem = ee.Image("USGS/SRTMGL1_003")
    img = dem.select("elevation").addBands(ee.Terrain.slope(dem).rename("slope_deg"))
    out = img.reduceRegions(collection=fields_fc(buffer_m=150),
                            reducer=ee.Reducer.mean(), scale=30)
    return to_df(out)


SOIL_PROPS = ["sand", "silt", "clay", "soc", "phh2o", "bdod"]


def run_soil():
    """SoilGrids 250 m (ISRIC community catalog). Raw units:
    sand/silt/clay g/kg, soc dg/kg, phh2o pH*10, bdod cg/cm3. Converted in features.py."""
    img = ee.Image.cat([ee.Image(f"projects/soilgrids-isric/{p}_mean") for p in SOIL_PROPS])
    out = img.reduceRegions(collection=fields_fc(), reducer=ee.Reducer.first(), scale=250)
    return to_df(out)


def run_cropmap():
    """EUCROPMAP crop class at each field for 2018 and 2022 (majority in a 60 m radius)."""
    ic = ee.ImageCollection("JRC/D5/EUCROPMAP/V1")
    rows, legend = [], None
    for year in (2018, 2022):
        img = ic.filterDate(f"{year}-01-01", f"{year + 1}-01-01").first().select("classification")
        out = img.reduceRegions(collection=fields_fc(buffer_m=60),
                                reducer=ee.Reducer.mode().setOutputs(["crop_code"]), scale=10)
        df = to_df(out)
        df["year"] = year
        rows.append(df)
        if legend is None:
            names = img.get("classification_class_names").getInfo()
            values = img.get("classification_class_values").getInfo()
            if names and values:
                legend = dict(zip([int(v) for v in values], names))
    df = pd.concat(rows, ignore_index=True)
    if legend:
        df["crop_name"] = df["crop_code"].map(lambda c: legend.get(int(c)) if pd.notna(c) else None)
        (RAW_GEE / "eucropmap_legend.json").write_text(json.dumps(legend, indent=2))
    return df


STATIC = {"terrain": run_terrain, "soilgrids": run_soil, "eucropmap": run_cropmap}


# ---------------------------------------------------------------- main
def save(name, df):
    path = RAW_GEE / f"{name}.csv"
    df.to_csv(path, index=False)
    span = ""
    if "date" in df:
        span = f"  {df['date'].min()} -> {df['date'].max()}"
    print(f"  OK  {name:15s} {len(df):6d} rows{span}  -> {path.name}")


def main():
    init_ee()
    failed = []

    for name, (cid, bands, scale, start, monthly) in SERIES.items():
        print(f"... {name}")
        try:
            save(name, run_series(name, cid, bands, scale, start, monthly))
        except Exception as e:
            failed.append(name)
            print(f"  FAIL {name}: {e}")
            traceback.print_exc(limit=1)

    for name, fn in STATIC.items():
        print(f"... {name}")
        try:
            save(name, fn())
        except Exception as e:
            failed.append(name)
            print(f"  FAIL {name}: {e}")
            traceback.print_exc(limit=1)

    print("\nDone." if not failed else f"\nDone with failures: {failed} (paste the error to Claude)")


if __name__ == "__main__":
    main()