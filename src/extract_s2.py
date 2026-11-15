"""
Step 1b: Sentinel-2 (10 m) NDVI for the 3 demo fields, for field-level bare-soil months.

Why: MODIS 250 m pixels mix each field with hedges and margins, so they almost never look bare.
Sentinel-2 at 10 m sees the field itself.

Run from the FARMS folder:
    python src/extract_s2.py

Output: data/raw/gee/s2_ndvi.csv  (one row per field per clear-enough image)
    ndvi_median    median NDVI of clear pixels within 40 m of the field point
    clear_frac     share of pixels in that circle that are not cloud / shadow / snow
    bare_scl_frac  share of clear pixels Sentinel-2 itself classes as "bare soil" (SCL = 5)
"""
import ee
import pandas as pd

from config import GEE_PROJECT, FIELDS, RAW_GEE, END_DATE

START = "2017-04-01"          # Sentinel-2 surface reflectance available for Europe from ~2017
BUFFER_M = 40
MIN_CLEAR = 0.5
BAD_SCL = [0, 1, 3, 7, 8, 9, 10, 11]   # no data, saturated, shadow, unclassified, clouds, cirrus, snow


def to_df(fc):
    try:
        return ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    except Exception:
        info = fc.getInfo()
        return pd.DataFrame([f["properties"] for f in info["features"]])


def main():
    ee.Initialize(project=GEE_PROJECT)
    print(f"Earth Engine ready (project: {GEE_PROJECT})")
    frames = []
    for f in FIELDS:
        circle = ee.Geometry.Point([f["lon"], f["lat"]]).buffer(BUFFER_M)
        ic = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(circle).filterDate(START, END_DATE))

        def per_image(img):
            scl = img.select("SCL")
            clear = scl.remap(BAD_SCL, [0] * len(BAD_SCL), 1).rename("clear")
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(clear)
            bare = scl.eq(5).rename("bare").updateMask(clear)
            # For a single-band image, reduceRegion names the result after the BAND, so the band is
            # renamed to "v". Fully cloudy images return an empty dictionary, so start from a default
            # that always has the key and overwrite it with the real value when there is one.
            def stat(image, reducer, default):
                real = image.rename("v").reduceRegion(reducer, circle, 10)
                return ee.Dictionary({"v": default}).combine(real, True).get("v")

            return ee.Feature(None, {
                "date": img.date().format("YYYY-MM-dd"),
                "ndvi_median": stat(ndvi, ee.Reducer.median(), -999),
                "bare_scl_frac": stat(bare, ee.Reducer.mean(), -999),
                "clear_frac": stat(clear, ee.Reducer.mean(), 0),
            })

        df = to_df(ee.FeatureCollection(ic.map(per_image)))
        df = df.drop(columns=[c for c in ("geo",) if c in df])
        df = df.replace(-999, float("nan"))
        df.insert(0, "field_id", f["id"])
        frames.append(df)
        kept = (df["clear_frac"] >= MIN_CLEAR).sum()
        print(f"  {f['id']}: {len(df)} images, {kept} clear enough (>= {MIN_CLEAR:.0%} clear)")

    out = pd.concat(frames, ignore_index=True)
    out = out[out["clear_frac"] >= MIN_CLEAR].dropna(subset=["ndvi_median"])
    out = out.sort_values(["field_id", "date"]).drop_duplicates(["field_id", "date"])
    path = RAW_GEE / "s2_ndvi.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} clear observations -> {path}")
    print(out.groupby("field_id")["date"].agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()