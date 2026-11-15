"""
Load the trained model and predict crop vigour for demo fields (any season features table).

Used by:
  - app/app.py      (climate scenarios: re-predict with changed weather)
  - add_field.py    (new fields: predict without retraining)
  - run directly:   python src/predict.py   -> rewrites data/processed/demo_predictions.csv for ALL fields
"""
import sys

import joblib
import pandas as pd

from config import MODELS, DATA_PROCESSED


def load_model():
    """train_model.py was run as a script, so its helper classes were pickled as __main__.X.
    Register them on whatever __main__ is now, then load."""
    import train_model
    main = sys.modules["__main__"]
    for name in ("LGBMWrap", "CropMean"):
        if not hasattr(main, name):
            setattr(main, name, getattr(train_model, name))
    return joblib.load(MODELS / "stress_model.joblib")


def predict(bundle, season_features):
    """season_features: rows of field_id, year + model features. Returns one row per field x year x crop."""
    feats = bundle["features"]
    rows = []
    for c in bundle["crop_codes"]:
        d = season_features.copy()
        d["crop"] = str(c)
        d["crop_code"] = c
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).dropna(subset=feats)
    X = d[["crop"] + feats]
    m = float(bundle.get("conformal_margin", 0.0))
    d["pred"] = bundle["model"].predict(X)
    d["pred_q10"] = bundle["q10"].predict(X) - m
    d["pred_q90"] = bundle["q90"].predict(X) + m
    keep = ["field_id", "year", "crop_code", "pred", "pred_q10", "pred_q90"] + \
           [c for c in ("ndvi_peak", "ndvi_mean_mj") if c in d]
    return d[keep].sort_values(["field_id", "crop_code", "year"]).reset_index(drop=True)


def main():
    bundle = load_model()
    demo = pd.read_csv(DATA_PROCESSED / "demo_season_features.csv")
    out = predict(bundle, demo)
    out.to_csv(DATA_PROCESSED / "demo_predictions.csv", index=False)
    print(f"Saved {len(out)} predictions for fields {sorted(out['field_id'].unique())}")


if __name__ == "__main__":
    main()