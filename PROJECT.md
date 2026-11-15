# FARMS: Field Shift for Lusatia

NASA Space Apps Challenge 2026, challenge "Field Shift: Adapting Farms with NASA Data".
Hackathon: 14 to 15 November 2026, local event Cottbus. Team: FARMS.

This file is the single source of truth for goals, decisions and progress.
Any agent working in this repo: read this first, and after finishing a step run
`python src/validate.py --step N` and only mark the step done when every check passes.

---

## 1. Goal

Build a local website that helps a farmer in Lusatia (around Cottbus, Germany) compare crop
rotation strategies using NASA Earth observations, soil data and their own priorities.

The tool must:
- show what NASA data says about the farmer's field (frost, heat, dry spells, soil moisture, bare-soil months, stress)
- compare at least 3 rotations, including the current one
- for each rotation: why it was suggested, the evidence, benefits, information gaps, and risks that need local review
- make uncertainty visible (range of outcomes over past seasons)
- support, not replace, farmer judgment and local expertise

Judging criteria (2025 guide, check the 2026 one): Impact, Creativity, Validity, Relevance, Presentation.
Validity is our main differentiator: real data, honest limitations, checks against ground data.

## 2. Scope

| Item | Decision |
|---|---|
| Region | Lusatia / southern Brandenburg, demo fields around Cottbus |
| ML training area | 5.9 to 23.0 E, 45.7 to 55.0 N (Germany + neighbours incl. warmer/drier analogs CZ, AT, PL, HU), Lusatia excluded |
| ML test area | Lusatia box 12.8 to 14.7 E, 51.2 to 51.95 N (held out, also the demo region) |
| Product | Streamlit website running locally (no cloud deployment) |
| Screens | Map -> My field -> My priorities -> Compare rotations, plus a Data & limitations panel |
| Everything precomputed | App reads files from data/processed and models/, works offline |

### Demo fields (checked on satellite imagery, all inside arable fields)

| ID | Lat | Lon | Place | EUCROPMAP 2018 | EUCROPMAP 2022 |
|---|---|---|---|---|---|
| field_south | 51.686380 | 14.254120 | Leuthen / Schorbus | Rapeseed | Common wheat |
| field_southeast | 51.694109 | 14.472812 | Near Komptendorf | Rye | Rye |
| field_west | 51.777853 | 14.178775 | Near Milkersdorf / Krieschow | Rapeseed | Maize |

## 3. Data sources

| Dataset | Access | Used for | Notes |
|---|---|---|---|
| NASA POWER daily (1981 to now) | API | Frost and heat calendar, dry spells, long-term trends (farmer view) | Solar radiation starts mid-1983 |
| SMAP L4 root-zone soil moisture | GEE `NASA/SMAP/SPL4SMGP/008` | Soil moisture trend, ML feature | ~9 km, 2015 onwards |
| GPM IMERG monthly | GEE `NASA/GPM_L3/IMERG_MONTHLY_V07` | Rainfall, ML feature | In GEE until ~Sept 2025 |
| MODIS NDVI / EVI | GEE `MODIS/061/MOD13Q1` | ML target, bare-soil months | 250 m, 16-day |
| MODIS LST | GEE `MODIS/061/MOD11A2` | Heat (farmer view) | 1 km, 8-day |
| MODIS ET / PET | GEE `MODIS/061/MOD16A2GF` | Water use (farmer view) | 500 m, 8-day |
| GLDAS NOAH 3-hourly | GEE `NASA/GLDAS/V021/NOAH/G025/T3H` | Heat days, late frost, GDD for ML (training and demo use the same function) | 0.25 deg |
| GRACE-FO mascon | GEE `NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI` | Regional groundwater trend (context only) | Very coarse, until Sept 2024 |
| Sentinel-2 SR (ESA) | GEE `COPERNICUS/S2_SR_HARMONIZED` | Field-level bare-soil months (3 demo fields) | 10 m, 2017 onwards, SCL cloud mask |
| SRTM | GEE `USGS/SRTMGL1_003` | Slope, erosion | 30 m |
| SoilGrids | GEE community catalog | Texture, organic carbon, pH; water holding via pedotransfer | 250 m, modelled snapshot |
| EUCROPMAP | GEE `JRC/D5/EUCROPMAP/V1` | Crop labels for ML | 2018 and 2022 only |
| ECOSTRESS ESI, ET, LST, SM | AppEEARS API | Field-scale stress snapshots, validation of the model | 70 m, irregular revisits, coverage only up to 52 N |
| DWD stations | DWD open data | Ground check of POWER temperature and rain | |
| County yields | Regionalstatistik | Sanity check of model (2018 should be a bad year) | |

## 4. Agreed decisions (log)

1. Field Shift chosen over astronaut health: real data, fewer local rivals, better fit.
2. Google Earth Engine is the pipeline; precompute everything, run locally.
3. Soil: static base (SoilGrids) + farmer override + dynamic signal from bare-soil months + estimated carbon direction per rotation.
4. ML model is a must-have; rule-based suitability score is the fallback.
5. ML target = peak NDVI (April to August) of that season, with crop type and site productivity
   (median peak NDVI 2003 to 2017, before training years, no leakage) as inputs.
   Reason: fields rotate crops, so a pixel-vs-own-average anomaly would mostly measure crop changes.
6. ML weather features come from GEE (IMERG, SMAP, GLDAS) with one shared function for training points
   and demo fields, so training and prediction inputs are identical. POWER stays for the farmer view.
7. Past-season replay covers 2015 to 2025 (SMAP starts 2015). Includes the 2018, 2019 and 2022 droughts.
8. ECOSTRESS is used for validation and field-scale stress, not as an ML input.
9. Left out on purpose: GRACE and MODIS LST as ML inputs, Landsat/Sentinel-2, management data,
   SMAP surface moisture and EVI (duplicates). Reasons in section 7.
10. POWER humidity, wind and radiation are used to compute FAO-56 reference ET0; crop water demand per
    rotation = crop coefficient x ET0 (farmer view, 3 demo fields). POWER GWETROOT gives a soil wetness
    trend back to 1981.
11. Terrain in the ML model: elevation, slope and topographic position (elevation minus 500 m mean).
    SHAP decides whether they matter.
12. ML water demand feature for training points = MODIS PET Jun-Aug (POWER ET0 needs one API call per point).
13. Training area is large (Germany + neighbours, incl. warmer/drier regions) so the model has seen weather
    Lusatia may face in future; Lusatia is held out as the test area and stays the demo region.
    Training points are sampled where >= 70% of the 250 m MODIS pixel is one crop.
14. Model benchmark: crop average, linear regression, MLP and LightGBM, all on the same spatial block CV.
    The best by spatial CV is used. Quantile models give a 10th to 90th percentile range.
15. ECOSTRESS QC, cloud and water layers are filters: bad readings are removed, raw files kept, and the
    share of removed readings is reported in the limitations panel.
16. SMAP soil moisture features: sm_rz_mj (May-July mean root-zone moisture) and sm_rz_anom_mj
    (that minus the same pixel's 2015-2025 May-July mean). Replaces sm_rootzone_pctl, which was
    missing for 34% of Lusatia points.
17. ECOSTRESS 2018-2019 summers and any summer with < 5 clear readings are marked low confidence;
    stress comparisons use 2020 onwards. The 2018 drought is shown with POWER, SMAP and NDVI instead.
18. Bare-soil months come from Sentinel-2 (10 m) for the demo fields (src/extract_s2.py). MODIS 250 m
    showed 0 to 0.8 bare months because pixels mix fields with hedges and winter months are cloudy.
19. Both targets (peak NDVI, mean May-July NDVI) are tried; the one where the full model adds most skill
    over crop + field quality on Lusatia is used (NDVI peak saturates and can hide stress).
20. Rotations use resilience: each crop's predicted drop in bad seasons vs its own typical season at that
    field, never raw NDVI compared across crops.
21. Spatial CV blocks ~100 km (1.4 x 0.9 deg), 5 folds, because weather inputs are coarse.
22. Year-transfer test (2018 <-> 2022) is mandatory and reported.
23. Light tuning only (LightGBM num_leaves 15/31/63, MLP 64-32 / 128-64), chosen on spatial CV.
24. Crop classes with < 100 rows are dropped; sorghum and millet use 'Other cereals' and are marked approximate.
    Final model is retrained on train + Lusatia; all reported scores come from before that.
    County yield check postponed to step 9 (only if time).
25. 10-90 % ranges are conformal-calibrated (split conformal on spatial CV out-of-fold errors) because the
    raw quantile models covered only 69 % of Lusatia points.
26. Current rotation = 2018 and 2022 EUCROPMAP crops cycled to 4 years (farmer edits it in the app).
    Alternatives = all 4-year cycles passing farming rules (harvest-to-sowing timing, no same family twice in a
    row except one cereal pair, rapeseed / potatoes / beet max 1 in 4, >= 1 legume, pH and sand fit).
    Cover crops are added automatically in gaps >= 3 months before a spring crop.
27. Without irrigation (default), crops with drought tolerance 1 (potatoes) are not proposed as alternatives.
28. Scores are 0-100 per component (see 29d for water and income). Sliders only re-weight, so ranking is instant.
29. Scorer fixes after reviewing the first real results (all fields got the same lucerne + millet + rapeseed
    alternatives, and alternatives increased the water deficit):
    a) lucerne / clover-grass only proposed if the farmer has a forage outlet (livestock / biogas), toggle off by default
    b) proso millet (niche market) only proposed if the farmer opts in
    c) weakly evidenced crops take the more cautious of model vs crop-table reliability
    d) water score is absolute: deficit compared with the field's own soil water store (mm, top 60 cm);
       income is absolute (0-1000 EUR/ha indicative). Decision 28 updated accordingly.
    e) alternatives = 'small step' (<= 2 of the 4 crops changed, built on the current rotation) and
       'bigger shift' (best overall, >= 2 crops different from both). Makes advice field-specific.
30. Reliability is soil-aware: on fields with < 70 mm soil water store, crop-table drought tolerance gets
    50 % weight (30 % otherwise) next to the ML resilience, because sandy soil cannot buffer dry spells.
31. App adds: glossary, 'satellite vs ground' and 'how crop types help soil' explainers, a harmonisation note,
    a heavy-rain card, and a vulnerability calendar (current crops' growth stages vs monthly frost, hot days and
    water gap) to cover the brief's 'seasons when current crops appear more vulnerable'.
32. Climate scenario sliders (summer rain -40..+20 %, warming 0..+3 C) re-run the same ML model and scorer on
    perturbed 2015-2025 seasons (src/scenario.py, delta method; heat and frost days recounted from POWER daily
    data). Labelled as a sensitivity test, not a projection. Scores show the change vs today's climate.
33. src/predict.py loads the trained model without retraining (registers the pickled helper classes).
34. src/add_field.py adds any field (id, lat, lon) and runs the pipeline for it (no retraining). Fields outside
    the training box are flagged as extrapolation in the app; ECOSTRESS only south of 52 N; current rotation
    from EUCROPMAP inside the EU, otherwise edited by the farmer. Extra fields live in data/fields_extra.json.
    validate.py checks now follow the number of fields.

## 5. Repository layout

```
FARMS/
  PROJECT.md            this file
  CLAUDE.md             short instructions for coding agents
  requirements.txt
  src/
    config.py           settings, fields, paths
    extract_gee.py      step 1
    extract_power.py    step 2
    appeears.py         step 3a (submit / status / download ECOSTRESS)
    load_ecostress.py   step 3b
    extract_training.py step 4
    features.py         step 5
    train_model.py      step 6
    rotations.py        step 7
    load_ecostress.py / extract_s2.py / features.py / train_model.py / rotations.py
    predict.py          load model + predict (scenarios, new fields)
    scenario.py         what-if climate perturbation
    add_field.py        add any field and run the pipeline
    validate_dwd.py     step 9
    validate.py         automated checks for every step
  notebooks/            exploration and charts
  data/raw/  data/processed/
  models/
  app/app.py            step 8
```

## 6. Steps and validation checkpoints

Each checkpoint is checked by `python src/validate.py --step N`. Manual checks are marked (manual).

### Step 1: GEE field data — DONE (validate passed)
Output: 9 CSVs in data/raw/gee/
- [ ] all 9 files exist: smap_monthly, imerg_monthly, modis_ndvi, modis_lst, modis_et, grace_mascon, terrain, soilgrids, eucropmap
- [ ] every time series has all 3 field_ids
- [ ] row counts: smap >= 350, imerg >= 800, ndvi >= 1500, lst >= 3000, et >= 3000, grace >= 500
- [ ] NDVI within -0.2..1, SMAP sm_rootzone within 0..0.9, LST day within -40..70 C, ET >= 0
- [ ] eucropmap: 6 rows, all crop codes in the arable range 200..299

### Step 2: NASA POWER daily — DONE (validate passed)
Output: data/raw/power/power_daily.csv
- [ ] 3 fields x >= 16,000 days, no duplicate (field, date)
- [ ] T2M_MIN <= T2M <= T2M_MAX on >= 99% of days
- [ ] precipitation >= 0
- [ ] mean annual rain 400..800 mm and mean temperature 7..12 C (Cottbus climate is about 560 mm, 9.5 C)
- [ ] 2018 April to August rain lower than the 1991 to 2020 average (known drought year)

### Step 3: ECOSTRESS — DONE (validate 10/10)
Output: AppEEARS CSVs in data/raw/ecostress/, cleaned file data/processed/ecostress_clean.csv
- [ ] AppEEARS task status = done, files downloaded
- [ ] at least one CSV per product (ESI, JET, LSTE, SM)
- [ ] all 3 field_ids present
- [ ] after cloud and quality filtering: >= 20 valid ESI observations per field
- [ ] ESI values within 0..1.5
- [x] (manual) 2022 shows lower ESI than 2021 and 2023. 2018 does NOT (highest ESI): mission started
      30 Jul 2018, so the 2018 summer is August after harvest, plus early instrument issues.
      Decision 17: ECOSTRESS 2018-2019 = low confidence.

### Step 4: ML training sample — NEXT
Command: `python src/extract_training.py --only test`, then `--only demo`, then full run.
Output: data/processed/training_sample.csv (region = train | test), data/processed/demo_season_features.csv
- [ ] years only 2018 and 2022
- [ ] Lusatia test set: >= 500 rows, all inside TEST_BOX
- [ ] train set: >= 5,000 rows, inside TRAIN_BOX, none inside Lusatia
- [ ] >= 6 crop classes with >= 100 rows each
- [ ] target ndvi_peak present in >= 95% of rows, values within 0.2..1.0
- [ ] all 18 feature columns present and complete in >= 95% of rows
- [ ] heat_days > 0 for most 2018 points (2018 was a hot summer)
- [ ] awc within 3..35 %
- [ ] demo_season_features: 3 fields x 11 seasons (2015..2025), all feature columns

### Step 5: Features and field profiles
Command: `python src/features.py`
Output: data/processed/field_profiles.json, monthly_field_table.csv, yearly_field_table.csv
- [ ] each field has: frost (last spring / first autumn, median and 1-in-10), heat days normal vs recent,
      rain, dry spells, FAO-56 ET0, water balance (rain - ET0), SMAP and POWER soil moisture trends,
      bare-soil months (Sentinel-2) and usually-bare calendar months, MODIS ET/PET, GRACE trend, soil texture + water holding, slope, crops, ECOSTRESS ESI
- [ ] last spring frost median between 15 March and 15 May
- [ ] water holding capacity within 5..30 % volume
- [ ] annual ET0 within 500..800 mm, annual rain within 400..800 mm
- [ ] monthly table 3 x 12 rows, yearly table covers 1981..2025
- [ ] trends reported per decade with p-value; only p < 0.05 called a trend in the app
- [ ] ECOSTRESS 2018-2019 and summers with < 5 readings marked low confidence
- [ ] (manual) notebook 01_explore shows plots for all datasets with no obvious errors

### Step 6: ML model
Command: `python src/train_model.py`
Output: models/stress_model.joblib, data/processed/model_report.json, demo_predictions.csv, figures/
- [ ] 5 models compared on ~100 km spatial block CV: crop average, crop + field quality, linear, MLP, LightGBM
- [ ] chosen model: CV R2 > 0, beats crop average AND crop + field quality (= NASA weather data adds skill)
- [ ] held-out Lusatia R2 > 0
- [ ] year transfer 2018 <-> 2022: model beats crop average both ways
- [ ] calibrated 10-90 % range covers 70-92 % of Lusatia test points (raw was 69 %)
- [ ] a water feature (rain, soil moisture, PET) in SHAP top 5
- [ ] demo_predictions.csv: 3 fields x 2015-2025 x every model crop
- [ ] (manual) Lusatia: model sees 2018 as worse than 2022, like the observations
- [ ] (optional, step 9) county yields check

### Step 7: Rotations
Command: `python src/rotations.py`
Output: data/crops.json, src/rotations.py, data/processed/rotations.json
- [ ] crops.json has >= 12 crops with water need, temperature limits, season, N-fixing, rooting depth, cover months, family, source
- [ ] scorer returns current + 'small step' (<= 2 crops changed) + 'bigger shift' for every demo field
- [ ] every strategy has: score 0..100, why, evidence, benefits, gaps, risks, check-locally list
- [ ] every strategy has a range over 2015..2025 (best, median, worst season)

### Step 8: App
Command: `streamlit run app/app.py`  (check: `python src/validate.py --step 8` runs it headless)
Output: app/app.py
- [ ] app starts with no exceptions, 4 tabs: Map, My field, Compare rotations, How it works & limits
- [ ] sidebar: field, current rotation (editable), 6 priority sliders, irrigation / forage / niche toggles,
      crops to exclude, soil-test override
- [ ] moving sliders changes the scores; all 3 fields load
- [ ] no Earth Engine calls (reads precomputed files only; map tiles need internet)
- [ ] My field: 6 cards incl. heavy rain, vulnerability calendar, 6 charts; How it works: glossary + explainers
- [ ] climate scenario sliders re-rank rotations and show the change vs today's climate
- [ ] a field added with add_field.py appears in the dropdown and loads
- [ ] (manual) a non-technical person can explain the result after 2 minutes

### Step 9: Ground checks and map layers
Output: data/processed/dwd_check.json, map layers in data/processed/maps/
- [ ] POWER vs DWD monthly temperature correlation >= 0.95, monthly rain correlation >= 0.6
- [ ] map layers exist for drought hotspots, bare-soil months, slope

### Step 10: Submission (manual)
- [ ] demo video recorded
- [ ] project page: NASA datasets used, method, limitations
- [ ] repo cleaned, README written
- [ ] submitted before the deadline, with buffer

## 7. Known limitations (shown in the app)

- Satellite data are estimates; checked against DWD stations and county yields where possible.
- Resolution: SMAP ~9 km, GLDAS ~25 km, GRACE ~300 km; field detail from MODIS (250 m) and ECOSTRESS (70 m).
- Crop labels only for 2018 and 2022; weather variety comes from many locations, not many years.
- Management (fertiliser, irrigation, sowing date) unknown.
- SoilGrids is a modelled snapshot; farmers can enter their own soil test.
- Datasets end on different dates (IMERG Sept 2025, GRACE Sept 2024).
- Market prices are indicative only.
- ECOSTRESS revisits are irregular and cloud-affected.