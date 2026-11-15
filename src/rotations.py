"""
Step 7: rotation scorer. Compares the current rotation with generated alternatives for each demo field.

Run from the FARMS folder:
    python src/rotations.py

The app imports this module: build_candidates() is computed once per field, rank() re-weights instantly
when the farmer moves the priority sliders.

Inputs : data/crops.json, data/processed/field_profiles.json, monthly_field_table.csv, demo_predictions.csv
Output : data/processed/rotations.json   (current + 2 alternatives per field, with explanations)

Six component scores (0-100, higher = better), weighted by farmer priorities:
  reliability  ML model: how much each crop's predicted vigour drops in the worst 2015-2025 seasons vs its own
               typical season at this field (resilience), blended with crop-table drought/heat tolerance
  water        crop water deficit vs effective rain (FAO-56 Kc x ET0), compared with THIS field's soil water
               store (mm in top 60 cm): sandy fields with little buffer are penalised more. Absolute score.
  soil_health  legumes, leys, deep roots, cover months (organic matter + infiltration), minus bare months and root-crop disturbance
               (an estimated DIRECTION of soil organic carbon, rules of thumb, not a measurement)
  erosion      bare months weighted by slope and sandy-soil wind erosion
  pest_break   crop family diversity, cereal share, same family within 2 years
  forage       livestock feed / biogas value of main crops (+ grazable cover crops), 0-5 table scale
  income       indicative gross margin (yield x price - variable cost), 0 = 0 EUR/ha, 100 = >= 1000 EUR/ha

Two kinds of alternative (fix e): "small_step" changes at most 2 of the 4 crops of the current rotation;
"bigger_shift" is the best overall rotation that differs from both.
"""
import itertools
import json

import numpy as np
import pandas as pd

from config import ROOT, DATA_PROCESSED, FIELDS

YEARS = 4
COMPONENTS = ["reliability", "water", "soil_health", "erosion", "pest_break", "income", "forage"]
DEFAULT_WEIGHTS = {k: 1.0 for k in COMPONENTS}
DEFAULT_WEIGHTS["forage"] = 0.0          # only matters to farms with livestock or a biogas outlet
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# EUCROPMAP class -> crop table id (used to guess the current rotation)
EUCROP_TO_CROP = {211: "winter_wheat", 213: "winter_barley", 214: "winter_rye", 215: "oats", 216: "maize",
                  218: "triticale", 219: "sorghum", 221: "potatoes", 222: "sugar_beet", 231: "sunflower",
                  232: "winter_rapeseed", 233: "soybean", 240: "field_pea", 250: "clover_grass"}
APPROX_ML = {"sorghum", "millet", "lupin", "clover_grass", "lucerne", "spring_barley", "field_pea"}


# ================================================================ loading
def load_inputs():
    crops = json.loads((ROOT / "data" / "crops.json").read_text())
    prof = json.loads((DATA_PROCESSED / "field_profiles.json").read_text())
    monthly = pd.read_csv(DATA_PROCESSED / "monthly_field_table.csv")
    preds = pd.read_csv(DATA_PROCESSED / "demo_predictions.csv")
    return crops, prof, monthly, preds


# ================================================================ calendar
def occupied_months(crop):
    """Months (as offsets relative to January of the harvest year) the crop is in the field.
    Winter crops start in autumn of the previous year (negative offsets)."""
    if crop["season"] == "perennial":
        return list(range(0, 12))
    sow, harv = min(crop["sow_months"]), max(crop["harvest_months"])
    start = sow - 12 if (crop["season"] == "winter" or sow > harv) else sow
    return list(range(start - 1, harv))  # 0-based month offsets, sow month .. harvest month


def feasible_pair(a, b):
    """Can crop b follow crop a (a harvested in year t, b harvested in year t+1)?"""
    if a["season"] == "perennial" or b["season"] == "perennial":
        return True
    a_end = max(a["harvest_months"])                    # month in year t
    if b["season"] == "winter" or min(b["sow_months"]) > max(b["harvest_months"]):
        return a_end <= max(b["sow_months"])            # b sown in autumn of year t
    return True                                         # spring/summer crop sown in year t+1


def timeline(seq, covers, add_covers=True):
    """Monthly cover over the 4 harvest years (48 months), cyclic. Returns covered flags + cover crops used.
    add_covers=False for the farmer's CURRENT rotation: we do not know that they sow cover crops, so none are
    assumed (Sentinel-2 bare months are the observed check). Alternatives include suggested cover crops."""
    covered = np.zeros(12 * YEARS, dtype=bool)
    for k, c in enumerate(seq):
        for m in occupied_months(c):
            covered[(12 * k + m) % (12 * YEARS)] = True
    used = []
    for k, c in enumerate(seq if add_covers else []):
        nxt = seq[(k + 1) % YEARS]
        if c["season"] == "perennial" or nxt["season"] in ("winter", "perennial"):
            continue
        harv = max(c["harvest_months"])                  # month index 1..12 in year k
        sow_next = min(nxt["sow_months"])                # in year k+1
        gap = (12 - harv) + (sow_next - 1)
        if gap < 3:
            continue
        fams = {c["family"], nxt["family"]}
        if nxt["id"] in ("maize", "sunflower", "sorghum", "soybean", "millet"):
            cc = covers["cc_vetch_rye"]; end = 12 + 3      # until end of March next year
        elif "Brassicaceae" in fams:
            cc = covers["cc_phacelia"]; end = 12            # winter-killed, covers to December
        else:
            cc = covers["cc_phacelia"]; end = 12
        for m in range(harv, end):
            covered[(12 * k + m) % (12 * YEARS)] = True
        used.append((k, cc["name"]))
    return covered, used


# ================================================================ per-crop field facts
def crop_resilience(preds, crop, fid):
    """From the ML replay 2015-2025: typical, worst and best predicted vigour, relative drop, years."""
    p = preds[(preds["field_id"] == fid) & (preds["crop_code"] == crop["ml_crop_code"])]
    if p.empty:
        return None
    typical = p["pred"].median()
    worst_row = p.loc[p["pred"].idxmin()]
    best_row = p.loc[p["pred"].idxmax()]
    return {"typical": float(typical), "worst": float(worst_row["pred"]), "worst_year": int(worst_row["year"]),
            "best": float(best_row["pred"]), "best_year": int(best_row["year"]),
            "worst_q10": float(worst_row["pred_q10"]),
            "drop_pct": float(100 * (typical - worst_row["pred"]) / typical),
            "by_year": {int(y): float(v) / typical for y, v in zip(p["year"], p["pred"])}}


def crop_stages(crop):
    """Month index (0 = Jan) -> stage for the months the crop is in the field, in growing order:
    1 = young / establishing / winter dormancy, 2 = main growth (highest water use), 3 = ripening.
    Winter crops: autumn establishment and Nov-Feb dormancy count as stage 1; main growth starts in March."""
    months = [m % 12 for m in occupied_months(crop)]
    if crop["season"] == "perennial":
        return {m: (2 if 3 <= m <= 8 else 1) for m in range(12)}
    harvest = max(crop["harvest_months"]) - 1
    stages = {}
    if crop["season"] == "winter" or min(crop["sow_months"]) > max(crop["harvest_months"]):
        wrapped = False                             # months come in growing order, starting at sowing
        for i, m in enumerate(months):
            if i > 0 and m < months[i - 1]:
                wrapped = True                      # crossed into the harvest year
            if not wrapped or m <= 1:
                stages[m] = 1                       # autumn establishment + winter dormancy (to end of Feb)
            elif m >= harvest - 1:
                stages[m] = 3                       # last two months before/at harvest = ripening
            else:
                stages[m] = 2                       # main growth from March
        return stages
    n = len(months)
    for i, m in enumerate(months):
        f = i / max(n - 1, 1)
        stages[m] = 1 if f < 0.3 else 3 if f > 0.85 else 2
    return stages


def crop_water(crop, et0, rain):
    """Seasonal crop water demand (mm) and deficit vs 80 % effective rain, from monthly normals.
    Kc follows the growth stage (FAO-56: initial / mid / end)."""
    kc_by_stage = {1: crop["kc"]["ini"], 2: crop["kc"]["mid"], 3: crop["kc"]["end"]}
    st_ = crop_stages(crop)
    demand = sum(kc_by_stage[v] * et0[m] for m, v in st_.items())
    deficit = sum(max(0.0, kc_by_stage[v] * et0[m] - 0.8 * rain[m]) for m, v in st_.items())
    return demand, deficit


# ================================================================ candidate generation + components
def rules_ok(seq, rules, ph, sand):
    """Farming rules for a 4-year cycle (the cycle wraps: year 4 is followed by year 1)."""
    ids = [c["id"] for c in seq]
    cereal_pairs = 0
    for k in range(YEARS):
        a, b = seq[k], seq[(k + 1) % YEARS]
        if not feasible_pair(a, b):                       # harvest of a must fit before sowing of b
            return False
        if a["family"] == b["family"]:
            if a["family"] == "Poaceae":
                cereal_pairs += 1                         # cereal after cereal: allowed once per cycle
            elif not (a["id"] == b["id"] and a["season"] == "perennial"):
                return False                              # e.g. pea after lupin, rapeseed after mustard family
    if cereal_pairs > 1:
        return False
    for cid in set(ids):                                  # perennial leys: max 2 years, other crops: max_share
        limit = 2 if any(c["id"] == cid and c["season"] == "perennial" for c in seq) else \
            max(1, int(rules["max_share"].get(cid, 0.5) * YEARS))
        if ids.count(cid) > limit:
            return False
    if rules.get("at_least_one_legume_in_4_years") and not any(c["n_fixing"] for c in seq):
        return False
    for c in seq:
        if c["id"] == "lupin" and ph is not None and ph > 6.8:
            return False
        if sand is not None and sand >= 70 and c["sandy_soil_fit"] <= 1:
            return False
    return sum(c["family"] == "Poaceae" for c in seq) <= 3


def components(seq, field, crop_info, covers, add_covers=True):
    covered, used_covers = timeline(seq, covers, add_covers)
    bare = 12 - covered.reshape(YEARS, 12).sum(axis=1).mean()
    soil = field["soil_terrain"]

    # reliability: ML resilience blended with crop-table tolerance. On fields with a small soil water store
    # (< 70 mm in the top 60 cm), drought tolerance matters more because the soil cannot buffer dry spells.
    store = float(soil.get("awc_mm_top_60cm") or 60)
    w_rule = 0.5 if store < 70 else 0.3
    rel_parts, approx = [], []
    for c in seq:
        info = crop_info[c["id"]]
        rule = 100 * (0.5 * c["drought_tolerance"] + 0.3 * c["heat_tolerance"] + 0.2 * c["sandy_soil_fit"]) / 5
        if info["res"] is not None:
            ml = float(np.clip(100 * (1 - info["res"]["drop_pct"] / 25), 0, 100))
            if c["id"] in APPROX_ML:
                rel_parts.append(min(ml, rule))           # weak evidence: take the more cautious of the two
            else:
                rel_parts.append((1 - w_rule) * ml + w_rule * rule)
        else:
            rel_parts.append(0.85 * rule)                 # no model evidence at all: cautious discount
        if c["id"] in APPROX_ML or info["res"] is None:
            approx.append(c["name"])
    reliability = float(np.mean(rel_parts))

    demand = float(np.mean([crop_info[c["id"]]["water"][0] for c in seq]))
    deficit = float(np.mean([crop_info[c["id"]]["water"][1] for c in seq]))

    legumes = sum(c["n_fixing"] for c in seq)
    perennial = sum(c["season"] == "perennial" for c in seq)
    deep = sum(c["max_root_depth_m"] >= 1.5 for c in seq)
    roots = sum(c["group"] == "root crop" for c in seq)
    soil_idx = 8 * legumes + 10 * perennial + 4 * deep + 2.5 * (12 - bare) - 8 * bare - 10 * roots
    soil_health = float(np.clip(50 + soil_idx - 25, 0, 100))

    slope = soil.get("slope_deg") or 0
    sand = soil.get("sand_pct") or 50
    erosion_risk = bare * (1 + slope / 4) * (1 + max(0, sand - 50) / 50) + 1.5 * sum(c["id"] in ("maize", "sunflower", "sorghum", "potatoes", "sugar_beet") for c in seq)
    erosion = float(np.clip(100 - 6 * erosion_risk, 0, 100))

    fams = [c["family"] for c in seq]
    same_within2 = sum(fams[k] == fams[(k + 2) % YEARS] for k in range(YEARS))
    cereal_share = fams.count("Poaceae") / YEARS
    pest = 100 * len(set(fams)) / YEARS - 10 * same_within2 - 30 * max(0, cereal_share - 0.5)
    pest_break = float(np.clip(pest, 0, 100))

    margins = [c["typical_yield"] * c["price_eur_per_unit"] - c["variable_cost_eur_ha"] for c in seq]
    gross = float(np.mean(margins))

    # forage: main crops on the 0-5 table scale; grazable cover crops add up to 15 points
    main_fv = float(np.mean([c.get("forage_value", 0) for c in seq])) / 5
    cover_by_name = {cc["name"]: cc for cc in covers.values()}
    cover_fv = sum(cover_by_name.get(n, {}).get("forage_value", 0) for _, n in used_covers) / 5
    forage = float(np.clip(100 * (0.85 * main_fv) + 15 * min(1.0, cover_fv / 2), 0, 100))

    return {"reliability": reliability, "demand_mm": demand, "deficit_mm": deficit, "soil_health": soil_health,
            "erosion": erosion, "pest_break": pest_break, "gross_eur_ha": gross, "bare_months": float(bare),
            "legumes": legumes, "covers": used_covers, "approx": sorted(set(approx)), "forage": forage}


def build_candidates(fid, crops, prof, monthly, preds, current_ids=None, exclude=(), irrigation=False,
                     forage_outlet=False, niche_ok=False):
    """Farmer constraints. Crops ruled out here are never proposed as alternatives (they stay scorable
    if they are part of the current rotation):
      exclude        crop ids the farmer cannot grow (machinery, market)
      irrigation     False -> crops with drought_tolerance 1 (potatoes) ruled out
      forage_outlet  False -> forage crops (lucerne, clover-grass) ruled out: they need livestock or biogas
      niche_ok       False -> niche-market crops (proso millet) ruled out"""
    field = prof[fid]
    all_crops = {c["id"]: c for c in crops["crops"]}
    ruled_out = set(exclude)
    if not irrigation:
        ruled_out |= {c["id"] for c in crops["crops"] if c["drought_tolerance"] <= 1}
    if not forage_outlet:
        ruled_out |= {c["id"] for c in crops["crops"] if c["group"] == "forage"}
    if not niche_ok:
        ruled_out |= {c["id"] for c in crops["crops"] if c.get("market") == "niche"}
    no_irr = ruled_out
    table = {cid: c for cid, c in all_crops.items() if cid not in ruled_out}
    covers = {c["id"]: c for c in crops["cover_crops"]}
    m = monthly[monthly["field_id"] == fid].set_index("month")
    et0 = [float(m.loc[i + 1, "et0_mm"]) for i in range(12)]
    rain = [float(m.loc[i + 1, "rain_mm"]) for i in range(12)]
    crop_info = {cid: {"res": crop_resilience(preds, c, fid), "water": crop_water(c, et0, rain)}
                 for cid, c in table.items()}
    ph = field["soil_terrain"].get("ph")
    sand = field["soil_terrain"].get("sand_pct")
    rules = crops["meta"]["rotation_rules"]

    if current_ids is None:
        cm = field.get("crops_eucropmap") or {}
        codes = [EUCROPMAP_CODE(n) for n in cm.values()]
        observed = [EUCROP_TO_CROP.get(c) for c in codes if EUCROP_TO_CROP.get(c) in all_crops]
        observed = observed or ["winter_rye"]
        current_ids = (observed * YEARS)[:YEARS]
    rows, seen = [], set()
    current = tuple(current_ids)
    for cid in current:                                   # current crops always scorable
        if cid not in table:
            table[cid] = all_crops[cid]
    crop_info.update({cid: {"res": crop_resilience(preds, all_crops[cid], fid),
                            "water": crop_water(all_crops[cid], et0, rain)} for cid in current if cid not in crop_info})
    alt_ids = [cid for cid in table if cid not in no_irr and cid not in set(exclude)]
    for seq_ids in itertools.chain([current], itertools.product(alt_ids, repeat=YEARS)):
        key = min(seq_ids[i:] + seq_ids[:i] for i in range(YEARS))   # same cycle, different start
        if key in seen:
            continue
        seq = [table[i] for i in seq_ids]
        is_current = seq_ids == current
        if not is_current and not rules_ok(seq, rules, ph, sand):
            continue
        seen.add(key)
        comp = components(seq, field, crop_info, covers, add_covers=not is_current)
        rows.append({"seq": seq_ids, "is_current": is_current, **comp})
    df = pd.DataFrame(rows)
    # water: deficit relative to this field's soil water store. ratio 1 -> 100, ratio 6 -> 0
    store = max(float(field["soil_terrain"].get("awc_mm_top_60cm") or 60), 20.0)
    df["deficit_to_store"] = df["deficit_mm"] / store
    df["water"] = 100 * ((6 - df["deficit_to_store"]) / 5).clip(0, 1)
    # income: absolute indicative gross margin, 0 EUR/ha -> 0, >= 1000 EUR/ha -> 100
    df["income"] = (df["gross_eur_ha"] / 10).clip(0, 100)
    return df, crop_info, {"et0": et0, "rain": rain, "soil_store_mm": store}


_NAME_TO_CODE = None


def EUCROPMAP_CODE(name):
    global _NAME_TO_CODE
    if _NAME_TO_CODE is None:
        p = DATA_PROCESSED.parent / "raw" / "gee" / "eucropmap_legend.json"
        legend = json.loads(p.read_text()) if p.exists() else {}
        _NAME_TO_CODE = {v: int(k) for k, v in legend.items()}
    return _NAME_TO_CODE.get(name)


def rank(df, weights=None):
    w = {k: float(v) for k, v in (weights or DEFAULT_WEIGHTS).items() if k in COMPONENTS}
    tot = sum(w.values()) or 1.0
    out = df.copy()
    out["score"] = sum(out[k] * v for k, v in w.items()) / tot
    return out.sort_values("score", ascending=False)


def changed(a, b):
    """How many of the 4 crops differ between two rotations (order-independent)."""
    from collections import Counter
    return sum((Counter(a) - Counter(b)).values())


def pick(ranked):
    """Current + 'small step' (best rotation changing at most 2 of the current crops) +
    'bigger shift' (best rotation overall that differs by >= 2 crops from both)."""
    cur = ranked[ranked["is_current"]].iloc[0]
    alts = ranked[~ranked["is_current"]]
    chosen = [("current", cur)]
    small = alts[alts["seq"].map(lambda s_: 1 <= changed(s_, cur["seq"]) <= 2)]
    if len(small):
        chosen.append(("small_step", small.iloc[0]))
    for _, r in alts.iterrows():
        if all(changed(r["seq"], c["seq"]) >= 2 for _, c in chosen):
            chosen.append(("bigger_shift", r))
            break
    return chosen


# ================================================================ explanations
def explain(r, cur, field, crop_info, table, fid):
    names = [table[i]["name"] for i in r["seq"]]
    cur_names = set(table[i]["name"] for i in cur["seq"])
    new = [n for n in names if n not in cur_names]
    w = field["water"]
    s2 = field.get("bare_soil_sentinel2") or {}
    soil = field["soil_terrain"]

    why, benefits = [], []
    d_def = cur["deficit_mm"] - r["deficit_mm"]
    if not r["is_current"]:
        if new:
            why.append(f"Brings in {', '.join(dict.fromkeys(new))}.")
        if d_def > 10:
            why.append(f"Needs about {d_def:.0f} mm less extra water per year than the current rotation.")
            benefits.append(f"Lower water deficit: {r['deficit_mm']:.0f} vs {cur['deficit_mm']:.0f} mm/yr.")
        if d_def < -10:
            why.append(f"Trade-off: needs about {-d_def:.0f} mm more extra water per year than the current rotation.")
        if r["bare_months"] < cur["bare_months"] - 0.3:
            why.append(f"Soil is bare {r['bare_months']:.1f} months/yr instead of {cur['bare_months']:.1f}.")
            benefits.append("More months of living soil cover (erosion, carbon, nutrient catch).")
        if r["legumes"] > table_legumes(cur, table):
            why.append("Adds a nitrogen-fixing legume.")
            benefits.append("Biological nitrogen reduces fertiliser need for the following crop.")
        if r["reliability"] > cur["reliability"] + 3:
            why.append("Its crops held up better in the worst recent seasons (model replay 2015-2025).")
        if r["pest_break"] > cur["pest_break"] + 5:
            benefits.append("More crop families: breaks pest and disease cycles.")
        if r["income"] > cur["income"] + 5:
            benefits.append("Higher indicative gross margin.")
        deep_new = [table[i]["name"] for i in dict.fromkeys(r["seq"])
                    if table[i]["max_root_depth_m"] >= 1.5 and i not in cur["seq"]]
        if deep_new or r["bare_months"] < cur["bare_months"] - 0.3:
            how = (f"deep roots of {', '.join(deep_new)} leave channels in the subsoil" if deep_new
                   else "longer living cover keeps the surface open and protected")
            benefits.append(f"Better infiltration: {how}, so heavy rain soaks in instead of running off.")
        if r["forage"] > cur["forage"] + 10:
            benefits.append("More feed for livestock or biogas (forage value "
                            f"{r['forage']:.0f} vs {cur['forage']:.0f}).")
    else:
        why.append("Your current rotation (assumed from the 2018 and 2022 satellite crop maps; edit it in the app).")
        if r["bare_months"] >= 2:
            seen = s2.get("bare_months_per_year")
            why.append(f"Scored as main crops only, without cover crops: about {r['bare_months']:.0f} bare months/yr. "
                       + (f"Sentinel-2 saw this field bare about {seen} months/yr, so if you already sow a winter "
                          "cover, your real bare time is lower." if seen is not None else
                          "If you already sow a winter cover, your real bare time is lower."))

    evidence = [
        f"Growing-season water balance (rain - ET0, Apr-Sep): {w['water_balance_apr_sep_normal_mm']} mm (1991-2020) -> "
        f"{w['water_balance_apr_sep_recent_mm']} mm (2011-2025). Source: NASA POWER, FAO-56.",
        f"Hot days > 30 C: {field['heat']['heat_days_30_normal']} -> {field['heat']['heat_days_30_recent']} per year (NASA POWER).",
        f"Soil: {soil['texture_class']}, water holding {soil['awc_pct_volume']} % (~{soil['awc_mm_top_60cm']} mm in top 60 cm), pH {soil['ph']} (SoilGrids).",
    ]
    if s2.get("bare_months_per_year") is not None:
        months = ", ".join(s2.get("usually_bare_months") or []) or "varies by year"
        evidence.append(f"Field currently bare {s2['bare_months_per_year']} months/yr, usually {months} (Sentinel-2, 10 m).")
    for cid in dict.fromkeys(r["seq"]):
        res = crop_info[cid]["res"]
        if res:
            evidence.append(f"{table[cid]['name']}: in its worst replay season ({res['worst_year']}) predicted vigour "
                            f"{res['drop_pct']:.0f} % below its typical season here (ML model on SMAP, IMERG, GLDAS, MODIS PET).")
    evidence.append(f"Water: average crop demand {r['demand_mm']:.0f} mm/yr, deficit vs rain {r['deficit_mm']:.0f} mm/yr (FAO-56 Kc x ET0).")

    gaps = ["Yields, prices and costs are indicative, not market data.",
            "Model trained on only 2 labelled years (2018, 2022); year-to-year transfer is weaker (R2 ~0.2-0.3).",
            "Soil values are modelled at 250 m (SoilGrids); a soil test overrides them."]
    if r["approx"]:
        gaps.append(f"Few or no training labels for: {', '.join(r['approx'])} (model result approximate, crop-table rules weigh more).")

    risks, check = [], ["Soil test (pH, texture, organic matter) for this field.",
                        "Market or contract for each new crop, and storage / drying.",
                        "Machinery for new crops (e.g. seed drill settings, harvest header)."]
    frost_1in10 = field["frost"]["last_spring_1_in_10"]
    for cid in dict.fromkeys(r["seq"]):
        c = table[cid]
        if c["late_frost_risk"] >= 4:
            risks.append(f"{c['name']}: frost-sensitive seedlings; here the last frost is after {frost_1in10} in 1 year in 10, so sow after that.")
        if soil.get("ph") is not None and soil["ph"] < c["ph_range"][0] - 0.2:
            risks.append(f"{c['name']}: prefers pH >= {c['ph_range'][0]}, field is ~{soil['ph']}: liming likely needed.")
        if c["drought_tolerance"] <= 2:
            risks.append(f"{c['name']}: drought-sensitive on light soil; yield risk in dry summers like 2018/2019/2022.")
        if cid == "winter_rapeseed":
            risks.append("Winter rapeseed needs moist soil at sowing in August, which has often been too dry recently.")
        if cid in ("sugar_beet", "potatoes"):
            check.append(f"{c['name']}: needs a processor contract and usually irrigation on sand.")
        if c["group"] == "forage":
            check.append(f"{c['name']}: needs a forage outlet (livestock or biogas).")
    if not r["is_current"] and d_def < -10:
        risks.append(f"Uses more water than your current rotation ({r['deficit_mm']:.0f} vs {cur['deficit_mm']:.0f} mm/yr "
                     "deficit): riskier in dry summers unless you can irrigate.")
    if r["covers"]:
        cc = sorted(set(n for _, n in r["covers"]))
        benefits.append(f"Cover crops fill bare gaps: {', '.join(cc)}.")
        check.append("Cover crop seed and sowing right after harvest.")
    check.append("Discuss with a local adviser (e.g. LELF Brandenburg / extension service) before changing.")

    seasons = season_range(r["seq"], crop_info)
    return {"why": why, "evidence": evidence, "benefits": benefits or ["Keeps the system you know."],
            "gaps": gaps, "risks": risks or ["No major crop-specific risk flagged."], "check_locally": check,
            "seasons": seasons}


def table_legumes(row, table):
    return sum(table[i]["n_fixing"] for i in row["seq"])


def season_range(seq, crop_info):
    """Rotation vigour index per replay season = mean over its crops of predicted / typical (x100)."""
    by_year = {}
    for cid in seq:
        res = crop_info[cid]["res"]
        if not res:
            continue
        for y, v in res["by_year"].items():
            by_year.setdefault(y, []).append(v)
    idx = {y: round(100 * float(np.mean(v)), 1) for y, v in sorted(by_year.items())}
    if not idx:
        return {"by_year": {}, "worst": None, "typical": None, "best": None}
    s = pd.Series(idx)
    return {"by_year": idx,
            "worst": {"year": int(s.idxmin()), "index": float(s.min())},
            "typical": {"index": round(float(s.median()), 1)},
            "best": {"year": int(s.idxmax()), "index": float(s.max())},
            "note": "100 = typical season for these crops at this field; 90 = 10 % lower vigour."}


KIND_LABEL = {"current": "Current rotation", "small_step": "Small step", "bigger_shift": "Bigger shift"}


def to_strategy(kind, r, cur, field, crop_info, table, fid):
    ex = explain(r, cur, field, crop_info, table, fid)
    if kind == "small_step":
        ex["why"].insert(0, f"Keeps {4 - changed(r['seq'], cur['seq'])} of your 4 crops and changes "
                            f"{changed(r['seq'], cur['seq'])}: an easy first step.")
    elif kind == "bigger_shift":
        ex["why"].insert(0, "The best-scoring rotation overall for your priorities: a larger change.")
    return {
        "kind": kind, "label": KIND_LABEL[kind],
        "crops_changed": int(changed(r["seq"], cur["seq"])),
        "name": " - ".join(table[i]["name"] for i in r["seq"]),
        "crops": list(r["seq"]),
        "is_current": bool(r["is_current"]),
        "score": round(float(r["score"]), 1),
        "components": {k: round(float(r[k]), 1) for k in COMPONENTS},
        "metrics": {"water_demand_mm": round(r["demand_mm"]), "water_deficit_mm": round(r["deficit_mm"]),
                    "bare_months": round(r["bare_months"], 1), "gross_margin_eur_ha_indicative": round(r["gross_eur_ha"]),
                    "deficit_vs_soil_store": round(float(r["deficit_to_store"]), 2),
                    "legumes": int(r["legumes"])},
        "cover_crops": [{"after_year": int(k) + 1, "crop": n} for k, n in r["covers"]],
        **ex,
    }


def run_field(fid, crops, prof, monthly, preds, weights=None, current_ids=None, exclude=(), irrigation=False,
              forage_outlet=False, niche_ok=False, candidates=None):
    """candidates: pass a cached build_candidates() result so slider changes only re-rank."""
    if candidates is None:
        candidates = build_candidates(fid, crops, prof, monthly, preds, current_ids, exclude, irrigation,
                                      forage_outlet, niche_ok)
    df, crop_info, clim = candidates
    ranked = rank(df, weights)
    chosen = pick(ranked)
    table = {c["id"]: c for c in crops["crops"]}
    cur = chosen[0][1]
    return {"n_candidates": int(len(df)), "soil_store_mm": clim["soil_store_mm"],
            "weights": weights or DEFAULT_WEIGHTS,
            "constraints": {"irrigation": irrigation, "forage_outlet": forage_outlet, "niche_ok": niche_ok,
                            "excluded": list(exclude)},
            "strategies": [to_strategy(k, r, cur, prof[fid], crop_info, table, fid) for k, r in chosen]}


def main():
    crops, prof, monthly, preds = load_inputs()
    out = {}
    for f in FIELDS:
        out[f["id"]] = run_field(f["id"], crops, prof, monthly, preds)
    (DATA_PROCESSED / "rotations.json").write_text(json.dumps(out, indent=2, default=str))

    for fid, r in out.items():
        print(f"\n=== {fid}  ({r['n_candidates']} rotations checked, soil water store {r['soil_store_mm']:.0f} mm)")
        for s in r["strategies"]:
            tag = {"current": "CURRENT ", "small_step": "SMALL   ", "bigger_shift": "BIGGER  "}[s["kind"]]
            c = s["components"]
            seas = s["seasons"]
            worst = f"worst {seas['worst']['year']}: {seas['worst']['index']}" if seas.get("worst") else ""
            print(f"  {tag} {s['score']:5.1f}  {s['name']}")
            print(f"          rel {c['reliability']:.0f} | water {c['water']:.0f} | soil {c['soil_health']:.0f} | "
                  f"erosion {c['erosion']:.0f} | pests {c['pest_break']:.0f} | income {c['income']:.0f} | forage {c['forage']:.0f}   "
                  f"deficit {s['metrics']['water_deficit_mm']} mm, bare {s['metrics']['bare_months']} mo, {worst}")
            if not s["is_current"]:
                for w in s["why"][:3]:
                    print(f"          - {w}")
    print("\nSaved data/processed/rotations.json")


if __name__ == "__main__":
    main()