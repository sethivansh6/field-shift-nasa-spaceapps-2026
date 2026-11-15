"""
Step 4: ML training sample + demo season features, all from Google Earth Engine.

One shared function, season_image(year), builds every feature. It is used for:
  - training points (2018, 2022) across Germany + neighbours  -> training data
  - Lusatia points (2018, 2022)                               -> held-out test data
  - the 3 demo fields (2015..2025)                            -> rotation replay
So training and prediction inputs are always computed the same way.

Run from the FARMS folder:
    python src/extract_training.py --only test     # Lusatia only, ~10-20 min (do this first)
    python src/extract_training.py --only demo     # 3 demo fields x 11 seasons, ~5 min
    python src/extract_training.py                 # everything incl. big train area, ~1-2 h

Resumable: finished batches are saved in data/processed/training_parts/ and skipped on rerun.
Outputs:
    data/processed/training_sample.csv       (region = train | test)
    data/processed/demo_season_features.csv
"""
import argparse
import json
import time

import ee
import numpy as np
import pandas as pd

from config import (GEE_PROJECT, FIELDS, DATA_PROCESSED, TEST_BOX, TRAIN_BOX, TILE_DEG,
                    POINTS_PER_CLASS_TILE, POINTS_PER_CLASS_TEST, PURITY_MIN,
                    TRAIN_YEARS, DEMO_YEARS, CROP_CODE_RANGE)

PARTS = DATA_PROCESSED / "training_parts"
PARTS.mkdir(parents=True, exist_ok=True)
BATCH = 200
SCALE = 250  # sample everything on the MODIS 250 m grid

FEATURES = ["site_ndvi", "precip_spring", "precip_summer", "sm_rz_mj", "sm_rz_anom_mj", "pet_summer",
            "heat_days", "late_frost_days", "gdd", "sand", "clay", "soc", "ph", "bdod",
            "elevation", "slope", "tpi"]
TARGETS = ["ndvi_peak", "ndvi_mean_mj"]


# ================================================================ helpers
def init_ee():
    ee.Initialize(project=GEE_PROJECT)
    print(f"Earth Engine ready (project: {GEE_PROJECT})")


def to_df(fc):
    try:
        return ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    except Exception:
        info = fc.getInfo()
        rows = []
        for f in info["features"]:
            r = dict(f["properties"])
            r["geo"] = f.get("geometry")
            rows.append(r)
        return pd.DataFrame(rows)


def date(y, m, d=1):
    return ee.Date.fromYMD(y, m, d)


def modis_proj():
    return ee.ImageCollection("MODIS/061/MOD13Q1").first().select("NDVI").projection()


# ================================================================ feature layers
def ndvi_ic():
    def prep(img):
        good = img.select("SummaryQA").lte(1)           # 0 good, 1 marginal
        return (img.select("NDVI").multiply(0.0001).updateMask(good)
                .set("system:time_start", img.get("system:time_start")))
    return ee.ImageCollection("MODIS/061/MOD13Q1").map(prep)


def site_ndvi():
    """Median peak NDVI 2003-2017: field quality measured before the training years."""
    years = ee.List.sequence(2003, 2017)
    peaks = years.map(lambda y: ndvi_ic().filterDate(date(y, 4), date(y, 9)).max())
    return ee.ImageCollection.fromImages(peaks).median().rename("site_ndvi")


def smap_mj(y):
    """Mean SMAP root-zone soil moisture (m3/m3), May-July of year y."""
    return (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").select("sm_rootzone")
            .filterDate(date(y, 5), date(y, 8)).mean())


def smap_mj_normal():
    """Same pixel's average May-July soil moisture over 2015-2025 (the 'normal')."""
    return ee.ImageCollection.fromImages(ee.List.sequence(2015, 2025).map(smap_mj)).mean()


def imerg_sum(y, m0, m1):
    """Total rain (mm) from month m0 to m1 inclusive."""
    ic = (ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").select("precipitation")
          .filterDate(date(y, m0), date(y, m1).advance(1, "month")))
    hours = lambda img: img.multiply(img.date().advance(1, "month").difference(img.date(), "hour"))
    return ic.map(hours).sum()


def gldas_daily(y):
    """Daily max/min/mean air temperature (C), 1 March to 31 August."""
    ic = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").select("Tair_f_inst")
    start = date(y, 3)
    n = date(y, 9).difference(start, "day")

    def day(i):
        d = start.advance(i, "day")
        sub = ic.filterDate(d, d.advance(1, "day"))
        img = ee.Image.cat([sub.max().rename("tmax"), sub.min().rename("tmin"),
                            sub.mean().rename("tmean")]).subtract(273.15)
        return img.set("system:time_start", d.millis())

    return ee.ImageCollection.fromImages(ee.List.sequence(0, n.subtract(1)).map(day))


def soil():
    """SoilGrids, depth-weighted mean 0-30 cm, converted to normal units."""
    spec = {"sand": ("sand", 10), "clay": ("clay", 10), "soc": ("soc", 100),   # %, %, %
            "ph": ("phh2o", 10), "bdod": ("bdod", 100)}                     # pH, g/cm3
    bands = []
    for name, (prop, div) in spec.items():
        img = ee.Image(f"projects/soilgrids-isric/{prop}_mean")
        v = (img.select(f"{prop}_0-5cm_mean").multiply(5)
             .add(img.select(f"{prop}_5-15cm_mean").multiply(10))
             .add(img.select(f"{prop}_15-30cm_mean").multiply(15))
             .divide(30 * div))
        bands.append(v.rename(name))
    return ee.Image.cat(bands)


def terrain():
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    tpi = dem.subtract(dem.focal_mean(500, "circle", "meters")).rename("tpi")
    return ee.Image.cat([dem.rename("elevation"), slope, tpi])


def season_image(y):
    """All targets and features for season y, as one multi-band image."""
    ndvi = ndvi_ic()
    daily = gldas_daily(y)
    bands = [
        ndvi.filterDate(date(y, 4), date(y, 9)).max().rename("ndvi_peak"),
        ndvi.filterDate(date(y, 5), date(y, 8)).mean().rename("ndvi_mean_mj"),
        site_ndvi(),
        imerg_sum(y, 3, 5).rename("precip_spring"),
        imerg_sum(y, 6, 8).rename("precip_summer"),
        smap_mj(y).rename("sm_rz_mj"),
        smap_mj(y).subtract(smap_mj_normal()).rename("sm_rz_anom_mj"),
        ee.ImageCollection("MODIS/061/MOD16A2GF").select("PET")
          .filterDate(date(y, 6), date(y, 9)).sum().multiply(0.1).rename("pet_summer"),
        daily.filterDate(date(y, 6), date(y, 9)).map(lambda i: i.select("tmax").gt(30))
          .sum().rename("heat_days"),
        daily.filterDate(date(y, 4, 15), date(y, 6)).map(lambda i: i.select("tmin").lt(0))
          .sum().rename("late_frost_days"),
        daily.filterDate(date(y, 3), date(y, 8)).map(lambda i: i.select("tmean").subtract(5).max(0))
          .sum().rename("gdd"),
        soil(),
        terrain(),
    ]
    return ee.Image.cat(bands)


def sample_image(img, fc):
    """Value of every band at every point; missing values stay as NaN (rows are not dropped)."""
    return to_df(img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=SCALE, tileScale=4))


# ================================================================ sampling points
def crop_layers(year):
    crop = (ee.ImageCollection("JRC/D5/EUCROPMAP/V1")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01").first().select("classification"))
    proj = modis_proj()
    mode = crop.reduceResolution(ee.Reducer.mode(), maxPixels=1024).reproject(proj)
    purity = crop.eq(mode).reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(proj)
    lo, hi = CROP_CODE_RANGE
    ok = purity.gte(PURITY_MIN).And(mode.gte(lo)).And(mode.lte(hi))
    return mode.rename("crop_code").toInt().updateMask(ok), crop


def tiles(box):
    w, s, e, n = box
    out = []
    lon = w
    while lon < e:
        lat = s
        while lat < n:
            out.append([lon, lat, min(lon + TILE_DEG, e), min(lat + TILE_DEG, n)])
            lat += TILE_DEG
        lon += TILE_DEG
    return out


def in_box(df, box):
    w, s, e, n = box
    return df["lon"].between(w, e) & df["lat"].between(s, n)


def sample_points(region, year):
    path = PARTS / f"points_{region}_{year}.csv"
    if path.exists():
        return pd.read_csv(path)
    masked, _ = crop_layers(year)
    boxes = [TEST_BOX] if region == "test" else tiles(TRAIN_BOX)
    per_class = POINTS_PER_CLASS_TEST if region == "test" else POINTS_PER_CLASS_TILE
    frames = []
    for k, b in enumerate(boxes):
        try:
            fc = masked.stratifiedSample(numPoints=per_class, classBand="crop_code",
                                         region=ee.Geometry.Rectangle(b), scale=SCALE,
                                         seed=year + k, geometries=True, tileScale=8)
            df = to_df(fc)
        except Exception as e:
            print(f"   tile {k} failed: {e}")
            continue
        if len(df):
            df["lon"] = df["geo"].map(lambda g: g["coordinates"][0])
            df["lat"] = df["geo"].map(lambda g: g["coordinates"][1])
            frames.append(df[["crop_code", "lon", "lat"]])
        print(f"   {region} {year} tile {k + 1}/{len(boxes)}: {len(df)} points")
    pts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["crop_code", "lon", "lat"])
    if region == "train":
        pts = pts[~in_box(pts, TEST_BOX)]          # keep Lusatia fully held out
    pts["year"] = year
    pts["region"] = region
    pts["point_id"] = [f"{region}_{year}_{i}" for i in range(len(pts))]
    pts.to_csv(path, index=False)
    return pts


# ================================================================ feature extraction
def extract_features(pts, region, year):
    img = season_image(year)
    n_batches = int(np.ceil(len(pts) / BATCH))
    for b in range(n_batches):
        out = PARTS / f"feat_{region}_{year}_{b:03d}.csv"
        if out.exists():
            continue
        chunk = pts.iloc[b * BATCH:(b + 1) * BATCH]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
            for r in chunk.itertuples()])
        for attempt in range(3):
            try:
                t0 = time.time()
                df = sample_image(img, fc)
                df.drop(columns=[c for c in ("geo",) if c in df]).to_csv(out, index=False)
                print(f"   {region} {year} batch {b + 1}/{n_batches} ok ({time.time() - t0:.0f}s)")
                break
            except Exception as e:
                print(f"   {region} {year} batch {b + 1} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))


def saxton_rawls_awc(sand, clay, soc):
    """Available water capacity (% volume) from texture and organic matter (Saxton & Rawls 2006)."""
    S, C, OM = sand / 100, clay / 100, soc * 1.724
    t1500 = -0.024*S + 0.487*C + 0.006*OM + 0.005*S*OM - 0.013*C*OM + 0.068*S*C + 0.031
    th1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251*S + 0.195*C + 0.011*OM + 0.006*S*OM - 0.027*C*OM + 0.452*S*C + 0.299
    th33 = t33 + (1.283 * t33**2 - 0.374 * t33 - 0.015)
    return (th33 - th1500) * 100


def merge_training():
    pts = pd.concat([pd.read_csv(p) for p in PARTS.glob("points_*.csv")], ignore_index=True)
    feats = [pd.read_csv(p) for p in PARTS.glob("feat_*.csv")]
    if not feats:
        print("No feature batches yet.")
        return
    feats = pd.concat(feats, ignore_index=True).drop_duplicates("point_id")
    df = pts.merge(feats, on="point_id", how="inner")
    legend_path = DATA_PROCESSED.parent / "raw" / "gee" / "eucropmap_legend.json"
    if legend_path.exists():
        legend = {int(k): v for k, v in json.loads(legend_path.read_text()).items()}
        df["crop_name"] = df["crop_code"].map(legend)
    df["awc"] = saxton_rawls_awc(df["sand"], df["clay"], df["soc"])
    cols = ["point_id", "region", "year", "lat", "lon", "crop_code", "crop_name"] + TARGETS + FEATURES + ["awc"]
    df = df[[c for c in cols if c in df]]
    out = DATA_PROCESSED / "training_sample.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    print(df.groupby(["region", "year"]).size().to_string())
    print("\nRows per crop:")
    print(df["crop_name"].fillna(df["crop_code"].astype(str)).value_counts().to_string())


# ================================================================ demo fields
def extract_demo():
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([f["lon"], f["lat"]]), {"field_id": f["id"]})
                               for f in FIELDS])
    rows = []
    for y in DEMO_YEARS:
        df = sample_image(season_image(y), fc)
        df["year"] = y
        rows.append(df.drop(columns=[c for c in ("geo",) if c in df]))
        print(f"   demo {y} ok")
    df = pd.concat(rows, ignore_index=True)
    df["awc"] = saxton_rawls_awc(df["sand"], df["clay"], df["soc"])
    out = DATA_PROCESSED / "demo_season_features.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["test", "train", "demo"], help="run one part only")
    a = ap.parse_args()
    init_ee()

    if a.only in (None, "demo"):
        print("... demo fields")
        extract_demo()
    regions = {None: ["test", "train"], "test": ["test"], "train": ["train"], "demo": []}[a.only]
    for region in regions:
        for year in TRAIN_YEARS:
            print(f"... sampling points: {region} {year}")
            pts = sample_points(region, year)
            print(f"    {len(pts)} points")
            extract_features(pts, region, year)
    if regions:
        merge_training()
    print("\nDone. Check with: python src/validate.py --step 4")


if __name__ == "__main__":
    main()