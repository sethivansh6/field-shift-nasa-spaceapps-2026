"""Shared settings for the FARMS project. Edit GEE_PROJECT before running anything."""
from pathlib import Path

# Your Google Cloud project registered for Earth Engine (noncommercial, Community tier)
GEE_PROJECT = "farms-spaceapps"  # <-- change to your project ID

# Folders
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RAW_GEE = DATA_RAW / "gee"
RAW_POWER = DATA_RAW / "power"
RAW_ECOSTRESS = DATA_RAW / "ecostress"
MODELS = ROOT / "models"

for p in (RAW_GEE, RAW_POWER, RAW_ECOSTRESS, DATA_PROCESSED, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# Demo fields (checked on satellite imagery: inside arable fields, all south of 52 N for ECOSTRESS)
FIELDS = [
    {"id": "field_south", "lat": 51.686380, "lon": 14.254120,
     "label": "Leuthen / Schorbus, south of Cottbus"},
    {"id": "field_southeast", "lat": 51.694109, "lon": 14.472812,
     "label": "Near Komptendorf, south-east of Cottbus"},
    {"id": "field_west", "lat": 51.777853, "lon": 14.178775,
     "label": "Near Milkersdorf / Krieschow, west of Cottbus"},
]

# Extra fields added with `python src/add_field.py ...` (kept in a small JSON file, not in code)
_EXTRA = ROOT / "data" / "fields_extra.json"
if _EXTRA.exists():
    import json as _json
    FIELDS = FIELDS + [f for f in _json.loads(_EXTRA.read_text()) if f["id"] not in {x["id"] for x in FIELDS}]

# Time window (each dataset also has its own start date in extract_gee.py)
END_DATE = "2026-11-01"

# ---------------------------------------------------------------- ML (step 4)
# Boxes are [west, south, east, north] in degrees
TEST_BOX = [12.8, 51.2, 14.7, 51.95]    # Lusatia: held-out test area + demo region
TRAIN_BOX = [5.9, 45.7, 23.0, 55.0]     # Germany + neighbours, incl. warmer/drier analogs (CZ, AT, PL, HU)
TILE_DEG = 2.0                          # train box is sampled tile by tile for spatial spread
POINTS_PER_CLASS_TILE = 15              # per crop class, per tile, per year
POINTS_PER_CLASS_TEST = 60              # per crop class in Lusatia, per year
PURITY_MIN = 0.7                        # share of the 250 m MODIS pixel covered by the same crop
TRAIN_YEARS = [2018, 2022]              # years with EUCROPMAP labels
DEMO_YEARS = list(range(2015, 2026))    # replay seasons (SMAP starts 2015)
CROP_CODE_RANGE = (211, 250)            # EUCROPMAP arable crop classes