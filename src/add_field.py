"""
Add a new field anywhere and run the whole pipeline for it (no model retraining needed).

Run from the FARMS folder:
    python src/add_field.py --id my_field --lat 51.62 --lon 14.05 --label "Near Drebkau"
    python src/add_field.py --id my_field --from-step features      # resume after a failed step
    python src/add_field.py --remove my_field                       # take it out again

Steps (all fields are refreshed; each prints its own progress):
    gee       Earth Engine time series (SMAP, IMERG, MODIS, GRACE, SRTM, SoilGrids, EUCROPMAP)
    power     NASA POWER daily weather 1981-now
    s2        Sentinel-2 bare-soil months
    seasons   season features 2015-2025 for the model (same function as training)
    features  field profiles (frost, heat, water balance, ET0, soil, ...)
    predict   model predictions for every crop and season
    rotations rotation strategies
Then restart the app; the new field appears in the Field dropdown.

Coverage checks:
    inside the training box (Germany + neighbours)  -> model results valid; outside -> marked extrapolation
    south of 52 N                                    -> ECOSTRESS could be added via appeears.py (optional)
    EU                                               -> current rotation guessed from EUCROPMAP; else edit it in the app
Note: while Earth Engine is in restricted mode (quota used), the gee / s2 / seasons steps are slower.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from config import ROOT, TRAIN_BOX  # noqa: E402

EXTRA = ROOT / "data" / "fields_extra.json"
STEPS = [
    ("gee", ["extract_gee.py"]),
    ("power", ["extract_power.py"]),
    ("s2", ["extract_s2.py"]),
    ("seasons", ["extract_training.py", "--only", "demo"]),
    ("features", ["features.py"]),
    ("predict", ["predict.py"]),
    ("rotations", ["rotations.py"]),
]


def load_extra():
    return json.loads(EXTRA.read_text()) if EXTRA.exists() else []


def save_extra(fields):
    EXTRA.write_text(json.dumps(fields, indent=2))


def coverage(lat, lon):
    w, s, e, n = TRAIN_BOX
    return {"in_training_area": bool(w <= lon <= e and s <= lat <= n),
            "ecostress_possible": bool(lat < 52.0)}


def run_steps(start):
    names = [n for n, _ in STEPS]
    if start not in names:
        sys.exit(f"Unknown step '{start}'. Steps: {names}")
    for name, cmd in STEPS[names.index(start):]:
        print(f"\n===== step: {name}  ({' '.join(cmd)})")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(SRC / cmd[0])] + cmd[1:], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"\nStep '{name}' failed. Fix the error above, then resume with:\n"
                     f"  python src/add_field.py --from-step {name}")
        print(f"===== {name} done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="short id, letters/digits/underscore, e.g. my_field")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--label", default="")
    ap.add_argument("--from-step", default=None, help="resume from this step")
    ap.add_argument("--remove", default=None, help="remove a previously added field id")
    a = ap.parse_args()

    if a.remove:
        fields = [f for f in load_extra() if f["id"] != a.remove]
        save_extra(fields)
        print(f"Removed {a.remove}. Re-run: python src/add_field.py --from-step features")
        return

    if a.from_step and not a.id:
        run_steps(a.from_step)
        return

    if not (a.id and a.lat is not None and a.lon is not None):
        ap.error("need --id, --lat and --lon (or --from-step / --remove)")
    if not a.id.replace("_", "").isalnum():
        ap.error("--id: use letters, digits and underscores only")
    from config import FIELDS
    if a.id in {f["id"] for f in FIELDS} and a.id not in {f["id"] for f in load_extra()}:
        ap.error(f"'{a.id}' is one of the built-in demo fields")

    cov = coverage(a.lat, a.lon)
    field = {"id": a.id, "lat": a.lat, "lon": a.lon,
             "label": a.label or f"Added field at {a.lat:.3f} N, {a.lon:.3f} E", **cov}
    fields = [f for f in load_extra() if f["id"] != a.id] + [field]
    save_extra(fields)

    print(f"Added {a.id} at {a.lat}, {a.lon}")
    print(f"  inside model training area (Germany + neighbours): {'yes' if cov['in_training_area'] else 'NO -> results are extrapolation'}")
    print(f"  ECOSTRESS coverage (south of 52 N): {'yes (optional: appeears.py)' if cov['ecostress_possible'] else 'no'}")
    print("  Tip: check on Google Maps satellite view that the point is inside a crop field, not on a road or forest.")
    run_steps(a.from_step or "gee")
    print(f"\nDone. Restart the app (Ctrl+C, then python -m streamlit run app/app.py) and pick '{a.id}'.")


if __name__ == "__main__":
    main()