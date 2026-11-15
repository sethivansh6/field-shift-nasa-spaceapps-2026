"""
FARMS: Field Shift for Lusatia. Local Streamlit website.

Run from the FARMS folder:
    streamlit run app/app.py

Reads only precomputed files (data/, models/). No Earth Engine calls. Map tiles need internet.
"""
import copy
import json
import sys
from pathlib import Path

import folium
import jinja2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rotations as R                                     # noqa: E402
import scenario as SC                                     # noqa: E402
from config import FIELDS, TEST_BOX, TRAIN_BOX, DATA_PROCESSED, RAW_GEE, RAW_POWER  # noqa: E402

st.set_page_config(page_title="FARMS: Field Shift for Lusatia", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

# full-width charts/tables on both old and new Streamlit versions
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _v >= (1, 46) else {"use_container_width": True}

FEATURE_LABEL = {
    "crop": "Crop type", "site_ndvi": "Field quality (2003-2017 greenness)", "precip_spring": "Spring rain (Mar-May)",
    "precip_summer": "Summer rain (Jun-Aug)", "sm_rz_mj": "Root-zone soil moisture (May-Jul)",
    "sm_rz_anom_mj": "Soil moisture vs normal (May-Jul)", "pet_summer": "Evaporative demand (Jun-Aug)",
    "heat_days": "Hot days > 30 C", "late_frost_days": "Late frost days", "gdd": "Season warmth (degree days)",
    "sand": "Sand %", "clay": "Clay %", "soc": "Organic carbon", "ph": "Soil pH", "bdod": "Soil density",
    "awc": "Soil water holding", "elevation": "Elevation", "slope": "Slope", "tpi": "Hollow vs ridge",
}
TARGET_LABEL = {"ndvi_mean_mj": "average crop greenness, May-July (NDVI)", "ndvi_peak": "peak crop greenness (NDVI)"}
COMP_LABEL = {"reliability": "Reliability in bad seasons", "water": "Water (vs your soil's store)",
              "soil_health": "Soil health", "erosion": "Erosion protection",
              "pest_break": "Pest & disease break", "income": "Income (indicative)",
              "forage": "Forage value (livestock / biogas)"}
KIND_COLOR = {"current": "#888780", "small_step": "#1D9E75", "bigger_shift": "#534AB7"}


# ================================================================ data
@st.cache_data
def load_all():
    p = DATA_PROCESSED
    d = {
        "profiles": json.loads((p / "field_profiles.json").read_text()),
        "monthly": pd.read_csv(p / "monthly_field_table.csv"),
        "yearly": pd.read_csv(p / "yearly_field_table.csv"),
        "preds": pd.read_csv(p / "demo_predictions.csv"),
        "crops": json.loads((ROOT / "data" / "crops.json").read_text()),
        "rotations": json.loads((p / "rotations.json").read_text()),
        "report": json.loads((p / "model_report.json").read_text()) if (p / "model_report.json").exists() else None,
        "qc": json.loads((p / "ecostress_qc_report.json").read_text()) if (p / "ecostress_qc_report.json").exists() else {},
    }
    ts = p / "training_sample.csv"
    d["test_points"] = pd.read_csv(ts).query("region == 'test'") if ts.exists() else pd.DataFrame()
    gp = RAW_GEE / "grace_mascon.csv"                      # GRACE/GRACE-FO monthly water storage -> yearly mean
    if gp.exists():
        g = pd.read_csv(gp, parse_dates=["date"]).dropna(subset=["lwe_thickness"])
        g["year"] = g["date"].dt.year
        n = g.groupby(["field_id", "year"])["lwe_thickness"].agg(["mean", "count"]).reset_index()
        n = n[n["count"] >= 6].rename(columns={"mean": "grace_lwe_cm"})[["field_id", "year", "grace_lwe_cm"]]
        d["yearly"] = d["yearly"].merge(n, on=["field_id", "year"], how="left")
    return d


@st.cache_resource
def model_bundle():
    import predict
    return predict.load_model()


@st.cache_data
def seasons_and_power():
    return pd.read_csv(DATA_PROCESSED / "demo_season_features.csv"), SC.load_power()


@st.cache_data(show_spinner="Re-running the model with the scenario weather...")
def scenario_inputs(dR, dT):
    """Model predictions and monthly climate for a what-if scenario (dR = summer rain change, dT = warming)."""
    D = load_all()
    if not dR and not dT:
        return D["preds"], D["monthly"]
    import predict
    seasons, power = seasons_and_power()
    preds = predict.predict(model_bundle(), SC.perturb_seasons(seasons, power, dR, dT))
    return preds, SC.perturb_monthly(D["monthly"], dR, dT)


@st.cache_data(show_spinner="Checking thousands of rotations against your field...")
def candidates(fid, current, irrigation, forage, niche, exclude, soil_override, dR=0.0, dT=0.0):
    D = load_all()
    preds, monthly = scenario_inputs(dR, dT)
    prof = copy.deepcopy(D["profiles"])
    if soil_override:
        so_ = dict(soil_override)
        st_ = prof[fid]["soil_terrain"]
        st_.update(so_)
        if "sand_pct" in so_:
            awc = R_awc(st_["sand_pct"], st_.get("clay_pct") or 5, st_.get("soc_pct") or 1)
            st_["awc_pct_volume"], st_["awc_mm_top_60cm"] = round(awc, 1), round(awc * 6)
    return R.build_candidates(fid, D["crops"], prof, monthly, preds, list(current), list(exclude),
                              irrigation, forage, niche), prof


def R_awc(sand, clay, soc):
    S, C, OM = sand / 100, clay / 100, soc * 1.724
    t1500 = -0.024*S + 0.487*C + 0.006*OM + 0.005*S*OM - 0.013*C*OM + 0.068*S*C + 0.031
    th1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251*S + 0.195*C + 0.011*OM + 0.006*S*OM - 0.027*C*OM + 0.452*S*C + 0.299
    th33 = t33 + (1.283 * t33**2 - 0.374 * t33 - 0.015)
    return (th33 - th1500) * 100


D = load_all()
CROPS = {c["id"]: c for c in D["crops"]["crops"]}
CROP_IDS = list(CROPS)
def _label(f):
    name = f["id"].replace("_", " ").title()
    lab = f.get("label") or ""
    return lab if lab.lower().startswith(name.lower()) else f"{name}: {lab}"   # no "Field X: Field X: ..."


FIELD_LABEL = {f["id"]: _label(f) for f in FIELDS}

# ================================================================ look & feel
CROP_ICON = {"winter_rye": "🌾", "winter_wheat": "🌾", "winter_barley": "🌾", "triticale": "🌾", "spring_barley": "🌾",
             "oats": "🌾", "maize": "🌽", "sunflower": "🌻", "winter_rapeseed": "🌼", "potatoes": "🥔",
             "sugar_beet": "🍠", "lupin": "🌸", "field_pea": "🫛", "soybean": "🫘", "lucerne": "🍀",
             "clover_grass": "☘️", "sorghum": "🌾", "millet": "🌾"}
def icon(cid): return CROP_ICON.get(cid, "🌱")
def chain(crops): return " → ".join(f"{icon(c)} {CROPS[c]['name'].split(' (')[0]}" for c in crops)

st.markdown("""<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #23401A 0%, #3B6D11 55%, #7A9A3A 100%); color: #F6F4EE;
       border-radius: 18px; padding: 22px 30px; margin-bottom: 18px;}
.hero h1 {color: #F6F4EE; font-size: 2.3rem; margin: 0; padding: 0;}
.hero p {color: #DCE8CF; font-size: 1.08rem; margin: 6px 0 0 0;}
.hero .tag {display:inline-block; background: rgba(255,255,255,.14); border-radius: 999px; padding: 3px 12px;
            font-size: .85rem; margin: 10px 6px 0 0; color: #F6F4EE;}
.step {font-weight: 700; color: #3B6D11; font-size: 1.05rem; margin-bottom: 2px;}
.best {background: #EEF4E4; border: 1px solid #C9DDB0; border-radius: 14px; padding: 12px 16px;}
.best .big {font-size: 2.1rem; font-weight: 800; color: #23401A; line-height: 1.1;}
.best .vs {color: #5F6B5A; font-size: .95rem;}
.best .crops {font-size: 1.02rem; margin: 6px 0 4px 0;}
.chips {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 6px 0 14px 0;}
.chip {border-radius: 14px; padding: 12px 14px; border: 1px solid #E4E0D4; background: #FFFFFF;}
.chip .t {font-size: .85rem; color: #5F6B5A;} .chip .v {font-size: 1.45rem; font-weight: 800; color: #1E2A22;}
.chip .s {font-size: .85rem; color: #3D4A3F; margin-top: 2px;}
.chip.g {border-left: 7px solid #1D9E75;} .chip.y {border-left: 7px solid #E0A526;} .chip.r {border-left: 7px solid #D2453A;}
.rothead {border-radius: 12px; padding: 10px 14px; color: #fff; margin-bottom: 6px;}
.rothead .n {font-size: 1.05rem; font-weight: 700;} .rothead .sc {font-size: 2rem; font-weight: 800; line-height: 1.1;}
.facts {font-size: .9rem; color: #3D4A3F; background: #F4F1E8; border-radius: 10px; padding: 8px 10px; margin: 6px 0;}
div[data-testid="stTabs"] button p {font-size: 1.05rem; font-weight: 600;}
div[data-testid="stButtonGroup"] > div {flex-wrap: wrap; overflow: visible; row-gap: 6px;}
</style>""", unsafe_allow_html=True)

# ================================================================ sidebar
GOALS = {   # one-tap presets for the 0-5 priority sliders
    "⚖️ Balanced": dict(reliability=3, water=3, soil_health=3, erosion=3, pest_break=3, income=3, forage=0),
    "💧 Water": dict(reliability=4, water=5, soil_health=2, erosion=2, pest_break=2, income=2, forage=0),
    "🌱 Soil": dict(reliability=2, water=3, soil_health=5, erosion=5, pest_break=4, income=2, forage=0),
    "💶 Income": dict(reliability=4, water=2, soil_health=2, erosion=2, pest_break=2, income=5, forage=0),
    "🐄 Livestock": dict(reliability=3, water=3, soil_health=3, erosion=3, pest_break=3, income=3, forage=5),
}
for k in COMP_LABEL:
    st.session_state.setdefault(f"w_{k}", GOALS["⚖️ Balanced"][k])
st.session_state.setdefault("goal", "⚖️ Balanced")
st.session_state.setdefault("forage", False)
if "fid" not in st.session_state:
    st.session_state.fid = FIELDS[0]["id"]


def _apply_goal():
    g = st.session_state.get("goal")
    if g in GOALS:
        for k, v in GOALS[g].items():
            st.session_state[f"w_{k}"] = v
        st.session_state["forage"] = g.startswith("🐄")


def _custom_goal():
    st.session_state["goal"] = None


st.sidebar.markdown("## 🌾 FARMS")
st.sidebar.caption("Field Shift for Lusatia: NASA data for crop rotation choices")
fid = st.session_state.fid
prof = D["profiles"][fid]

default_cur = D["rotations"][fid]["strategies"][0]["crops"]
with st.sidebar.expander("🔄 Your current rotation", expanded=False):
    cur = []
    for i in range(4):
        cur.append(st.selectbox(f"Year {i + 1}", CROP_IDS, index=CROP_IDS.index(default_cur[i]),
                                format_func=lambda x: f"{icon(x)} {CROPS[x]['name']}", key=f"cur_{fid}_{i}"))
    st.caption("Pre-filled from the 2018 and 2022 EU satellite crop maps. Change it to match your farm.")

with st.sidebar.expander("🎚️ Fine-tune priorities (0 = ignore, 5 = very important)", expanded=False):
    weights = {k: st.slider(v, 0, 5, key=f"w_{k}", on_change=_custom_goal,
                            help=("Feed for your own animals or crops sold to a biogas plant. Leave at 0 if "
                                  "you have neither.") if k == "forage" else None)
               for k, v in COMP_LABEL.items()}

with st.sidebar.expander("🚜 Your farm", expanded=False):
    irrigation = st.checkbox("I can irrigate", False)
    forage = st.checkbox("I have livestock or a biogas plant (forage crops OK)", key="forage")
    niche = st.checkbox("Niche markets OK (e.g. proso millet)", False)
    exclude = st.multiselect("Crops I cannot grow (machinery, market)", CROP_IDS,
                             format_func=lambda x: f"{icon(x)} {CROPS[x]['name']}")

with st.sidebar.expander("🧪 I have a soil test"):
    use_test = st.checkbox("Use my soil test instead of SoilGrids")
    sg = prof["soil_terrain"]
    ph_in = st.number_input("pH", 3.5, 8.5, float(sg.get("ph") or 6.0), 0.1)
    sand_in = st.number_input("Sand %", 0.0, 100.0, float(sg.get("sand_pct") or 70.0), 1.0)
soil_override = (("ph", ph_in), ("sand_pct", sand_in)) if use_test else ()

with st.sidebar.expander("🌦️ Climate what-if", expanded=True):
    dR = st.slider("Summer rain change (Jun-Aug)", -40, 20, 0, 5, format="%d%%", key="sc_rain") / 100
    dT = st.slider("Warmer by", 0.0, 3.0, 0.0, 0.5, format="+%.1f C", key="sc_temp")
    if (dR or dT) and not (RAW_POWER / "power_daily.csv").exists():   # e.g. a deployment without the raw weather file
        st.error("Climate scenario unavailable here: the NASA POWER weather file "
                 "(data/raw/power/power_daily.csv) is missing. Showing today's climate.")
        dR, dT = 0.0, 0.0
    st.caption("Replays 2015-2025 with changed weather through the same model. A sensitivity test, "
               "not a climate projection.")
scenario_on = bool(dR or dT)
if scenario_on:
    st.sidebar.warning(f"Scenario on: summer rain {dR:+.0%}, +{dT:.1f} C. Rotation scores and replay use the "
                       "changed weather; My field shows observed data.")

(cand, prof_used) = candidates(fid, tuple(cur), irrigation, forage, niche, tuple(exclude), soil_override, dR, dT)
preds_used, monthly_used = scenario_inputs(dR, dT)
result = R.run_field(fid, D["crops"], prof_used, monthly_used, preds_used, weights=weights,
                     candidates=cand, irrigation=irrigation, forage_outlet=forage, niche_ok=niche,
                     exclude=tuple(exclude))
base_scores = {}
if scenario_on:      # same rotations scored in today's climate, to show the scenario effect
    (cand0, _) = candidates(fid, tuple(cur), irrigation, forage, niche, tuple(exclude), soil_override)
    r0 = R.rank(cand0[0], weights)
    base_scores = {tuple(sq): float(sc_) for sq, sc_ in zip(r0["seq"], r0["score"])}
prof = prof_used[fid]

st.sidebar.divider()
st.sidebar.caption("Supports, does not replace, your judgement and local advisers.")

# ================================================================ header + 3 steps
st.markdown("""<div class="hero"><h1>🌾 FARMS: Field Shift for Lusatia</h1>
<p>Which crop rotation will hold up in drier, hotter summers? NASA satellite data, checked for your field.</p>
<span class="tag">🛰️ 8 NASA datasets</span><span class="tag">🤖 ML trained on 13,000+ fields</span>
<span class="tag">📍 Real fields in Lusatia</span></div>""", unsafe_allow_html=True)

s1, s2, s3 = st.columns([1, 1.25, 1.2], gap="large")
with s1:
    st.markdown('<div class="step">① Pick your field</div>', unsafe_allow_html=True)
    st.selectbox("Field", [f["id"] for f in FIELDS], format_func=lambda x: FIELD_LABEL[x], key="fid",
                 label_visibility="collapsed")
    seen = prof.get("crops_eucropmap") or {}
    st.caption(f"🟫 {prof['soil_terrain']['texture_class']} soil  ·  🛰️ seen from space: "
               + ", ".join(f"{y}: {n}" for y, n in seen.items()))
    runs = []
    for c in cur:
        if runs and runs[-1][0] == c:
            runs[-1][1] += 1
        else:
            runs.append([c, 1])
    st.caption("🔄 Your rotation now: " + " → ".join(f"{icon(c)} {CROPS[c]['name'].split(' (')[0]}"
                                                  + (f" ×{n}" if n > 1 else "") for c, n in runs)
               + "  (edit in the sidebar)")
with s2:
    st.markdown('<div class="step">② What matters most to you?</div>', unsafe_allow_html=True)
    st.pills("Goal", list(GOALS), key="goal", on_change=_apply_goal, label_visibility="collapsed")
    st.caption("💧 save water · 🌱 healthy soil · 💶 keep income · 🐄 livestock or biogas. One tap sets your "
               "priorities; fine-tune each one in the sidebar"
               + (" (now: custom)." if not st.session_state.get("goal") else "."))
strategies = result["strategies"]
with s3:
    st.markdown('<div class="step">③ Your best option</div>', unsafe_allow_html=True)
    cur_s = next(s for s in strategies if s["kind"] == "current")
    best = max((s for s in strategies if s["kind"] != "current"), key=lambda s: s["score"])
    bm, cm = best["metrics"], cur_s["metrics"]
    st.markdown(f"""<div class="best"><div class="big">{best['score']:.0f} <span class="vs">points ·
{best['label']}</span></div><div class="vs">Your current rotation scores {cur_s['score']:.0f}</div>
<div class="crops">{chain(best['crops'])}</div>
<div class="vs">💧 water gap {bm['water_deficit_mm']} vs {cm['water_deficit_mm']} mm/yr · 🟫 bare soil
{bm['bare_months']} vs {cm['bare_months']} months/yr</div></div>""", unsafe_allow_html=True)
    st.caption("Why, risks and what to check: see **🔄 Compare rotations** below.")

_f = next(f for f in FIELDS if f["id"] == fid)
if not (TRAIN_BOX[0] <= _f["lon"] <= TRAIN_BOX[2] and TRAIN_BOX[1] <= _f["lat"] <= TRAIN_BOX[3]):
    st.error("This field is outside the model's training area (Germany + neighbours): model-based results are "
             "extrapolation. Climate, soil and satellite facts are still valid.")

tab_rot, tab_field, tab_map, tab_how = st.tabs(["🔄 Compare rotations", "🌱 My field", "🗺️ Stress map",
                                                "📖 How it works & limits"])

# ================================================================ tab 1: map
with tab_map:
    # layer -> (column, label, unit, True if LOW values mean stress, fixed (red, orange) cut-offs or None = thirds)
    LAYERS = {
        "Crop greenness vs normal (MODIS NDVI)": ("anom", "greenness vs crop average", "NDVI", True, (-0.05, -0.01)),
        "Soil moisture vs normal (NASA SMAP)": ("sm_rz_anom_mj", "root-zone soil moisture vs 2015-2025, May-Jul",
                                                "m3/m3", True, (-0.02, -0.005)),
        "Summer rain (NASA GPM IMERG)": ("precip_summer", "rain Jun-Aug", "mm", True, None),
        "Hot days > 30 C (NASA GLDAS)": ("heat_days", "days > 30 C in the season", "days", False, None),
        "Evaporative demand (NASA MODIS)": ("pet_summer", "potential evapotranspiration Jun-Aug", "mm", False, None),
        "Late frost nights (NASA GLDAS)": ("late_frost_days", "nights below 0 C between 15 Apr and 31 May",
                                           "nights", False, (1.5, 0.5)),
    }
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("**Stress hotspots**")
        layer = st.radio("Layer", list(LAYERS), key="map_layer")
        year = st.radio("Season", [2018, 2022], horizontal=True, key="map_year")
        col_, lab_, unit_, low_bad, fixed = LAYERS[layer]
    tp = D["test_points"]
    cuts = None
    if len(tp):
        tp = tp.copy()
        tp["anom"] = tp["ndvi_mean_mj"] - tp.groupby("crop_code")["ndvi_mean_mj"].transform("mean")
        vals = tp[col_].dropna()
        if fixed:                                            # anomalies: fixed, physically meaningful cut-offs
            cuts = fixed if low_bad else (fixed[1], fixed[0])
        else:                                                # absolute values: thirds of both seasons together
            cuts = (vals.quantile(1 / 3), vals.quantile(2 / 3))
    with c2:
        if cuts is not None:
            lo, hi = cuts
            bad, good = ((f"below {lo:.3g}", f"above {hi:.3g}") if low_bad else (f"above {hi:.3g}", f"below {lo:.3g}"))
            kind = ("Red = clear stress" if fixed else "Red = most stressed third")
            st.caption(f"Each dot is a crop field point in the held-out Lusatia test area: {lab_} ({unit_}). "
                       f"{kind} ({bad}), orange = in between, green = {'little or no stress' if fixed else 'least stressed'} "
                       f"({good}). Cut-offs are the same for both seasons, so 2018 and 2022 can be compared directly.")
            sub = tp[tp["year"] == year][col_].dropna()
            if len(sub):
                share = ((sub < lo) if low_bad else (sub > hi)).mean()
                st.metric(f"Points in the red class, {year}", f"{share:.0%}")
                st.caption(f"Season average: {sub.mean():.3g} {unit_}")
        st.caption("Squares = the demo fields (darker = the one selected). Box = Lusatia test area (never used to train the model).")
    with c1:
        lats, lons = [f["lat"] for f in FIELDS], [f["lon"] for f in FIELDS]
        m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=10, tiles="OpenStreetMap",
                       scrollWheelZoom=False)
        _bb = [[min(lats) - 0.05, min(lons) - 0.08], [max(lats) + 0.05, max(lons) + 0.08]]      # all fields in view
        # The map sits in a tab that is hidden on first load, so Leaflet starts with size 0. Fit the view once the
        # map is actually visible (fit_bounds at build time would zoom out to the whole world).
        fit = folium.MacroElement()
        fit._template = jinja2.Template(
            "{% macro script(this, kwargs) %}(function() { var _m = " + m.get_name() + "; var _t = setInterval("
            "function() { if (_m.getSize().x > 50) { _m.invalidateSize(); _m.fitBounds(" + json.dumps(_bb) + ");"
            " clearInterval(_t); } }, 200); })();{% endmacro %}")
        fit.add_to(m)
        w_, s_, e_, n_ = TEST_BOX
        folium.Rectangle([[s_, w_], [n_, e_]], color="#534AB7", fill=False, weight=1).add_to(m)
        if cuts is not None:
            lo, hi = cuts
            for r in tp[tp["year"] == year].itertuples():
                v = getattr(r, col_)
                if pd.isna(v):
                    continue
                stress = (v < lo, v < hi) if low_bad else (v > hi, v > lo)
                col = "#A32D2D" if stress[0] else "#EF9F27" if stress[1] else "#639922"
                folium.CircleMarker([r.lat, r.lon], radius=3, color=col, fill=True, fill_opacity=0.8, weight=0,
                                    tooltip=f"{lab_}: {v:.3g} {unit_}").add_to(m)
        for f in FIELDS:
            folium.RegularPolygonMarker([f["lat"], f["lon"]], number_of_sides=4, radius=9,
                                        color="#042C53" if f["id"] == fid else "#185FA5", fill=True,
                                        fill_opacity=0.9, tooltip=FIELD_LABEL[f["id"]]).add_to(m)
        st_folium(m, height=520, use_container_width=True, returned_objects=[], key=f"map_{layer}_{year}")

    st.subheader("Trends over time at the demo fields")
    TRENDS = {
        "Hot days > 30 C": ("heat_days_30", "days per year", "NASA POWER"),
        "Water balance Apr-Sep (rain - ET0)": ("water_balance_apr_sep_mm", "mm", "NASA POWER, FAO-56"),
        "Rain Apr-Aug": ("rain_apr_aug_mm", "mm", "NASA POWER"),
        "Longest dry spell Apr-Aug": ("longest_dry_spell_apr_aug", "days", "NASA POWER"),
        "Heavy rain days (>= 20 mm)": ("heavy_rain_days_20mm", "days per year", "NASA POWER"),
        "Last spring frost": ("last_spring_frost_doy", "day of year", "NASA POWER"),
        "Root-zone soil moisture Apr-Sep": ("smap_sm_apr_sep", "m3/m3", "NASA SMAP"),
        "Summer land surface temperature": ("summer_lst_day_c", "C", "NASA MODIS"),
        "Water storage vs 2004-2009 (GRACE/GRACE-FO)": ("grace_lwe_cm", "cm of water", "NASA GRACE / GRACE-FO, ~300 km"),
    }
    yt = D["yearly"]
    avail = [k for k, v in TRENDS.items() if v[0] in yt.columns and yt[v[0]].notna().sum() > 5]
    tsel = st.selectbox("Show", avail, key="trend_var")
    tcol, tunit, tsrc = TRENDS[tsel]
    ty = yt.dropna(subset=[tcol]).copy()
    ty = ty[ty["year"] <= 2025]
    # insert empty years so lines break at data gaps (e.g. 2017-2018 between GRACE and GRACE-FO)
    yrs = range(int(ty["year"].min()), int(ty["year"].max()) + 1)
    ty = (ty.set_index(["field_id", "year"])[[tcol]]
            .reindex(pd.MultiIndex.from_product([ty["field_id"].unique(), yrs], names=["field_id", "year"]))
            .reset_index())
    ty["field"] = ty["field_id"].map(FIELD_LABEL).fillna(ty["field_id"])
    same_cell = ty.groupby("year")[tcol].nunique(dropna=True).max() <= 1
    ft = px.line(ty, x="year", y=tcol, color="field", markers=True)
    avg = ty.groupby("year")[tcol].mean().rolling(10, min_periods=5).mean()
    ft.add_scatter(x=avg.index, y=avg.values, name="10-year average (all fields)",
                   line=dict(color="#2C2C2A", width=3, dash="dot"))
    ft.update_layout(height=380, yaxis_title=tunit, xaxis_title="", margin=dict(t=20, b=10),
                     legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(ft, **WIDE)
    first, last = avg.dropna().iloc[0] if avg.notna().any() else None, avg.dropna().iloc[-1] if avg.notna().any() else None
    if first is not None:
        if tcol.endswith("_doy"):
            fmt = lambda d: (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(round(d)) - 1)).strftime("%d %b")
            first_s, last_s, tunit_s = fmt(first), fmt(last), ""
        else:
            first_s, last_s, tunit_s = f"{first:.3g}", f"{last:.3g}", tunit
        gaps = [y for y in yrs if ty.loc[ty["year"] == y, tcol].isna().all()]
        note = (" All fields fall in the same data cell, so their lines lie on top of each other."
                if same_cell else "")
        note += (f" No data for {', '.join(map(str, gaps))}." if gaps else "")
        st.caption(f"{tsel}: 10-year average went from {first_s} to {last_s} {tunit_s} "
                   f"({int(avg.dropna().index[0])} to {int(avg.dropna().index[-1])}). Source: {tsrc}.{note}")

# ================================================================ tab 2: my field
with tab_field:
    fr, ht, wa, so = prof["frost"], prof["heat"], prof["water"], prof["soil_terrain"]
    s2 = prof.get("bare_soil_sentinel2") or {}
    def _lvl(v, g, r, low_bad=False):           # traffic light: green / yellow / red
        if v is None:
            return "y"
        return ("g" if v >= g else "r" if v < r else "y") if low_bad else ("g" if v < g else "r" if v >= r else "y")
    _doy = lambda d: pd.Timestamp(f"2001 {d}").dayofyear if d else None
    hot, hot0 = ht["heat_days_30_recent"], ht["heat_days_30_normal"]
    wb, wb0 = wa["water_balance_apr_sep_recent_mm"], wa["water_balance_apr_sep_normal_mm"]
    bare, heavy = s2.get("bare_months_per_year"), wa.get("heavy_rain_days_recent")
    chips = [
        ("❄️ Late spring frost", fr["last_spring_median"], _lvl(_doy(fr["last_spring_1_in_10"]), 116, 126),
         f"1 year in 10 as late as {fr['last_spring_1_in_10']}. Matters for early sowing.",
         "NASA POWER 1991-2020. Green before 26 Apr, red from 6 May (1-in-10 date)."),
        ("🔥 Hot days > 30 °C", f"{hot} /yr", _lvl(hot, 8, 15),
         f"{hot - hot0:+.1f} vs 1991-2020 ({hot0}).", "NASA POWER. Green under 8, red from 15 per year."),
        ("💧 Water gap Apr-Sep", f"{-wb:.0f} mm", _lvl(wb, -150, -250, low_bad=True),
         f"The air takes more water than rain gives. Gap {wb0 - wb:+.0f} mm vs 1991-2020.",
         "Rain minus FAO-56 reference evapotranspiration (NASA POWER). Green under 150 mm gap, red over 250 mm."),
        ("🪣 Soil water store", f"{so['awc_mm_top_60cm']:.0f} mm", _lvl(so["awc_mm_top_60cm"], 100, 60, low_bad=True),
         f"{so['texture_class']}: roots can draw on this much.",
         f"SoilGrids, top 60 cm, {so['awc_pct_volume']} % available water. Green from 100 mm, red under 60 mm."),
        ("🟫 Bare soil", f"{bare if bare is not None else '-'} months/yr", _lvl(bare, 1, 3),
         "Usually bare: " + (", ".join(s2.get("usually_bare_months") or []) or "varies") + ".",
         "Sentinel-2 10 m, months with field NDVI < 0.3. Green under 1, red over 3."),
        ("🌧️ Heavy rain", f"{heavy if heavy is not None else '-'} days/yr", _lvl(heavy, 1, 3),
         "Days over 20 mm: wash topsoil off bare, sandy fields.",
         "NASA POWER 2011-2025. Green under 1, red from 3 per year."),
    ]
    st.markdown("#### Your field at a glance")
    st.markdown('<div class="chips">' + "".join(
        f'<div class="chip {c}" title="{h}"><div class="t">{t}</div><div class="v">{v}</div><div class="s">{x}</div></div>'
        for t, v, c, x, h in chips) + "</div>", unsafe_allow_html=True)
    st.caption("🟢 fine  ·  🟡 watch  ·  🔴 a real risk for crops here. Hover a card for the source and cut-offs.")

    st.markdown(f"**In plain words:** in a typical growing season this field loses about "
                f"{-wa['water_balance_apr_sep_recent_mm']:.0f} mm more water to the air than falls as rain, while the soil "
                f"can only store about {so['awc_mm_top_60cm']:.0f} mm for roots. Hot days have gone from "
                f"{ht['heat_days_30_normal']} to {ht['heat_days_30_recent']} a year. Crops that root deep, finish early "
                f"or tolerate drought carry less risk here.")

    mon = D["monthly"].query("field_id == @fid")
    yr = D["yearly"].query("field_id == @fid")

    st.subheader("When are your current crops most exposed?")
    risk = pd.DataFrame({
        "Frost days": mon["frost_days"].values,
        "Hot days > 30 C": mon["heat_days_30_recent"].values,
        "Water gap (ET0 - rain, mm)": (mon["et0_mm"] - mon["rain_mm"]).clip(lower=0).values,
    }, index=R.MONTHS)
    crop_rows, notes = {}, []
    for cid in dict.fromkeys(cur):
        c = CROPS[cid]
        stage = R.crop_stages(c)                 # 1 young / winter dormancy, 2 main growth, 3 ripening
        crop_rows[c["name"]] = [stage.get(m, 0) for m in range(12)]
        main = [m for m, v in stage.items() if v == 2]          # in growing order (winter crops span New Year)
        gap = float(risk["Water gap (ET0 - rain, mm)"].iloc[main].sum()) if main else 0.0
        hot = float(risk["Hot days > 30 C"].iloc[main].sum()) if main else 0.0
        young = [m for m, v in stage.items() if v == 1 and 2 <= m <= 5]
        frost = float(risk["Frost days"].iloc[young].sum()) if young and c["late_frost_risk"] >= 3 else 0.0
        worst = R.MONTHS[max(main, key=lambda m: risk["Water gap (ET0 - rain, mm)"].iloc[m])] if main else "-"
        txt = (f"**{c['name']}**: main growth {R.MONTHS[main[0]]}-{R.MONTHS[main[-1]]} faces a water gap of "
               f"~{gap:.0f} mm and ~{hot:.0f} hot days (worst month: {worst})") if main else f"**{c['name']}**"
        if frost >= 1:
            txt += f"; frost-sensitive seedlings meet ~{frost:.0f} frost days in spring"
        if c["drought_tolerance"] <= 2 and gap > 100:
            txt += ". Drought-sensitive crop in the driest part of the year: the most vulnerable season."
        notes.append(txt)
    z = pd.DataFrame(crop_rows, index=R.MONTHS).T
    fc1 = go.Figure()
    fc1.add_heatmap(z=z.values, x=R.MONTHS, y=list(z.index), colorscale=[[0, "#F1EFE8"], [0.33, "#C0DD97"],
                    [0.66, "#3B6D11"], [1, "#BA7517"]], zmin=0, zmax=3, showscale=False,
                    hovertemplate="%{y} %{x}<extra></extra>")
    fc1.update_layout(height=60 + 40 * len(z), margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
    fr_ = go.Figure()
    for col_, colr in zip(risk.columns, ["#378ADD", "#D85A30", "#BA7517"]):
        fr_.add_bar(x=R.MONTHS, y=risk[col_], name=col_, marker_color=colr)
    fr_.update_layout(height=230, margin=dict(t=10, b=10), barmode="group", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fc1, **WIDE)
    st.caption("Crop calendar of your current rotation: light green = young crop or winter dormancy, dark green = "
               "main growth (highest water need), brown = ripening, grey = not in the field.")
    st.plotly_chart(fr_, **WIDE)
    for t_ in notes:
        st.markdown("- " + t_)

    a, b = st.columns(2)
    with a:
        f1 = go.Figure()
        f1.add_bar(x=R.MONTHS, y=mon["rain_mm"], name="Rain", marker_color="#378ADD")
        f1.add_bar(x=R.MONTHS, y=mon["et0_mm"], name="Water demand of the air (ET0)", marker_color="#EF9F27")
        f1.update_layout(title="Where the water gap is (monthly normal 1991-2020)", barmode="group",
                         height=330, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f1, **WIDE)
        st.caption("Source: NASA POWER daily data, FAO-56 Penman-Monteith ET0.")
    with b:
        y2 = yr[yr["year"] >= 1985]
        f2 = go.Figure()
        f2.add_bar(x=y2["year"], y=y2["heat_days_30"], name="Hot days > 30 C", marker_color="#D85A30", opacity=0.6)
        f2.add_scatter(x=y2["year"], y=y2["heat_days_30"].rolling(10, min_periods=5).mean(), name="10-year average",
                       line=dict(color="#712B13", width=3))
        f2.update_layout(title="Hot days per year", height=330, margin=dict(t=50, b=10),
                         legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f2, **WIDE)
        tr = ht.get("heat_days_trend") or {}
        if tr:
            st.caption(f"Trend {tr['per_decade']:+.1f} days per decade "
                       f"({'statistically significant' if tr['significant'] else 'not significant'}, p = {tr['p_value']}). NASA POWER.")
    c, d_ = st.columns(2)
    with c:
        f3 = go.Figure()
        f3.add_bar(x=yr["year"], y=yr["water_balance_apr_sep_mm"], marker_color=np.where(yr["water_balance_apr_sep_mm"] < 0, "#E24B4A", "#378ADD"))
        f3.update_layout(title="Growing-season water balance (rain - ET0, Apr-Sep), mm", height=320, margin=dict(t=50, b=10))
        st.plotly_chart(f3, **WIDE)
    with d_:
        if "smap_sm_apr_sep" in yr:
            sm = yr.dropna(subset=["smap_sm_apr_sep"])
            f4 = px.line(sm, x="year", y="smap_sm_apr_sep", markers=True,
                         title="Root-zone soil moisture, Apr-Sep (NASA SMAP)")
            f4.update_layout(height=320, margin=dict(t=50, b=10), yaxis_title="m3/m3")
            st.plotly_chart(f4, **WIDE)
    e1, e2 = st.columns(2)
    with e1:
        es = yr.dropna(subset=["ecostress_esi_summer"]) if "ecostress_esi_summer" in yr else yr.iloc[0:0]
        if es.empty:
            st.markdown("**Summer water stress (ECOSTRESS)**")
            st.caption("No ECOSTRESS data downloaded for this field. It is optional: request it with "
                       "src/appeears.py (south of 52 N only). All scores and recommendations work without it.")
        else:
            f5 = px.bar(es, x="year", y="ecostress_esi_summer", color="ecostress_confidence",
                        color_discrete_map={"ok": "#1D9E75", "low": "#B4B2A9"},
                        title="Summer water stress (ECOSTRESS ESI, 70 m; lower = more stress)")
            f5.update_layout(height=320, margin=dict(t=50, b=10), legend_title="confidence")
            st.plotly_chart(f5, **WIDE)
            st.caption("2018-2019 low confidence: mission start late July 2018 and early instrument issues.")
    with e2:
        if "s2_bare_share" in mon:
            f6 = px.bar(x=R.MONTHS, y=mon["s2_bare_share"] * 100, title="How often the field is bare, by month (Sentinel-2, %)")
            f6.update_traces(marker_color="#BA7517")
            f6.update_layout(height=320, margin=dict(t=50, b=10), yaxis_title="% of years bare", xaxis_title="")
            st.plotly_chart(f6, **WIDE)
            st.caption("Bare months are where a cover crop protects soil from wind erosion and keeps nutrients.")
    g = prof.get("groundwater_grace") or {}
    st.info(f"Regional water storage (GRACE-FO, ~300 km): {g.get('trend_cm_per_year')} cm per year over "
            f"{g.get('years')}. Context only, not field-level.")

# ================================================================ tab 3: compare rotations
with tab_rot:
    st.markdown(f"Checked **{result['n_candidates']:,}** rotations that pass the farming rules for this field. "
                "Scores are 0-100, combined from your priorities (step ②).")
    if scenario_on:
        st.info(f"Climate scenario on: summer rain {dR:+.0%}, temperature +{dT:.1f} C. "
                "Scores below use the changed weather.")
    cols = st.columns(len(strategies), gap="medium")
    for col, s in zip(cols, strategies):
        with col, st.container(border=True):
            sub = ""
            if scenario_on and tuple(s["crops"]) in base_scores:
                sub = f"<div>{s['score'] - base_scores[tuple(s['crops'])]:+.1f} points vs today's climate</div>"
            st.markdown(f"<div class='rothead' style='background:{KIND_COLOR[s['kind']]}'>"
                        f"<div class='n'>{s['label']}</div><div class='sc'>{s['score']:.0f} / 100</div>{sub}</div>",
                        unsafe_allow_html=True)
            for i, cid in enumerate(s["crops"]):
                cc = [c["crop"] for c in s["cover_crops"] if c["after_year"] == i + 1]
                st.markdown(f"Year {i + 1}: {icon(cid)} **{CROPS[cid]['name']}**"
                            + (f"  \n  🌱 then cover: {cc[0]}" if cc else ""))
            mtr = s["metrics"]
            st.markdown(f"<div class='facts'>💧 water gap <b>{mtr['water_deficit_mm']} mm/yr</b> · 🟫 bare "
                        f"<b>{mtr['bare_months']} months/yr</b><br>🫛 legumes <b>{mtr['legumes']}</b> · 💶 margin "
                        f"<b>~{mtr['gross_margin_eur_ha_indicative']} EUR/ha</b> (indicative)</div>",
                        unsafe_allow_html=True)
            with st.expander("💡 Why", expanded=True):
                for x in s["why"]:
                    st.markdown(f"- {x}")
            comp = pd.DataFrame({"goal": [COMP_LABEL[k] for k in s["components"]], "score": list(s["components"].values())})
            fc = px.bar(comp, x="score", y="goal", orientation="h", range_x=[0, 100])
            fc.update_traces(marker_color=KIND_COLOR[s["kind"]])
            fc.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.markdown("**Score by goal**")
            st.plotly_chart(fc, key=f"comp_{s['kind']}", config={"displayModeBar": False}, **WIDE)
            for ttl, key_ in (("✅ Benefits", "benefits"), ("⚠️ Risks", "risks"), ("🔍 Check locally", "check_locally"),
                              ("📊 Evidence", "evidence"), ("❓ Information gaps", "gaps")):
                with st.expander(ttl):
                    for x in s[key_]:
                        st.markdown(f"- {x}")

    st.subheader("How each rotation would have done in the real weather of 2015-2025")
    rows = []
    for s in strategies:
        for y, v in (s["seasons"].get("by_year") or {}).items():
            rows.append({"year": int(y), "index": v, "rotation": s["label"]})
    if rows:
        sr = pd.DataFrame(rows)
        fs = px.line(sr, x="year", y="index", color="rotation", markers=True,
                     color_discrete_map={s["label"]: KIND_COLOR[s["kind"]] for s in strategies})
        fs.add_hline(y=100, line_dash="dot", line_color="#888780")
        fs.update_layout(height=360, yaxis_title="crop vigour vs typical season (100 = typical)", margin=dict(t=20))
        st.plotly_chart(fs, **WIDE)
        st.caption("Predicted by the ML model from each season's NASA weather (SMAP soil moisture, IMERG rain, GLDAS "
                   "temperature, MODIS evaporative demand) and this field's soil. Below 100 = a weaker season for those "
                   "crops. Model ranges are calibrated to cover ~80 % of real outcomes; year-to-year accuracy is "
                   "moderate, so read this as relative risk, not a yield forecast.")

# ================================================================ tab 4: how it works
with tab_how:
    rep = D["report"]
    st.subheader("The model")
    if rep:
        t = rep["chosen_target"]
        comp = rep["comparison"][t]
        tab = pd.DataFrame([{"model": n, "spatial CV R2": v["spatial_cv"]["r2"], "Lusatia R2 (held out)": v["lusatia"]["r2"]}
                            for n, v in comp.items()])
        tab["model"] = tab["model"].map({"crop_mean": "Crop average", "crop_site": "Crop + field quality (no weather)",
                                         "linear": "Linear regression", "mlp": "Neural network (MLP)",
                                         "lgbm": "Gradient-boosted trees (LightGBM)"})
        a, b = st.columns([1, 1])
        with a:
            st.dataframe(tab, hide_index=True, **WIDE)
            st.caption("R2 = share of variation explained on areas the model never saw (1 = perfect, 0 = no better than "
                       "guessing the average; negative = worse than guessing the average). The simple baselines fail on "
                       "Lusatia because its sandy fields differ from the training average; the weather-aware models do not.")
            st.markdown(f"- Predicts **{TARGET_LABEL.get(t, t)}** from season weather + soil + crop.\n"
                        f"- NASA weather data lifts R2 from **{rep['crop_site_r2']}** (crop + field only) to "
                        f"**{rep['cv_r2']}** (~100 km spatial cross-validation).\n"
                        f"- Year transfer (train one year, test the other): "
                        + ", ".join(f"{k} R2 {v['model']['r2']}" for k, v in rep["year_transfer"].items()) +
                        " (weaker: only 2 labelled years).\n"
                        f"- 10-90 % ranges calibrated to cover **{rep['quantile_coverage_lusatia']:.0%}** of held-out Lusatia points.\n"
                        f"- Trained on {rep['data']['rows_used']:,} crop field points across Germany and neighbours; "
                        "Lusatia held out for testing.")
        with b:
            imp = rep.get("shap_importance") or {}
            if imp:
                si = pd.Series(imp).sort_values(ascending=True).tail(12)
                fsh = go.Figure(go.Bar(x=si.values, y=[FEATURE_LABEL.get(i, i) for i in si.index], orientation="h",
                                       marker_color="#3B6D11"))
                fsh.update_layout(title="What drives crop vigour in Lusatia", height=420, margin=dict(t=50, b=10),
                                  xaxis_title="average effect on predicted greenness (SHAP)")
                st.plotly_chart(fsh, **WIDE)
                st.caption("SHAP shows how much each input moves the model's prediction, on held-out Lusatia fields. "
                           "Summer rain and evaporative demand matter most after crop type: water drives vigour here.")

    st.subheader("How the data are combined")
    st.markdown("Every dataset has a different resolution (from 10 m Sentinel-2 to ~300 km GRACE) and timing "
                "(3-hourly to yearly). For the model, all are **harmonised to one analysis grid**: values are taken "
                "on the 250 m MODIS grid and summarised per season (e.g. June-August rain, May-July mean soil "
                "moisture). Coarse data (weather, soil moisture) describe the conditions; fine data (MODIS, "
                "Sentinel-2, ECOSTRESS) describe the field itself.")

    st.subheader("Glossary")
    with st.expander("Terms used in this app", expanded=False):
        st.markdown("""
- **Evapotranspiration (ET):** water leaving the field as evaporation from soil plus transpiration from plants.
- **Reference ET (ET0):** how much water a short green grass would use under today's weather (sun, heat, wind,
  dry air). A measure of how "thirsty" the air is. Computed with the FAO-56 Penman-Monteith method from NASA POWER.
- **Crop coefficient (Kc):** a crop's water use relative to ET0 at each growth stage (e.g. maize ~1.2 at peak).
- **Water balance:** rain minus ET0. Negative = the air demands more water than falls; crops live on stored soil water.
- **Soil water store (available water capacity):** how much water the soil can hold for roots. Sand holds little.
- **Infiltration:** how fast rain soaks into the soil instead of running off. Deep roots (lucerne, sunflower, rye)
  leave channels, and living cover protects the surface from crusting. FARMS counts both in the soil-health score.
- **Soil moisture anomaly:** this season's soil moisture minus the usual value for the same place and months.
- **NDVI:** satellite greenness index (0 = bare soil, ~0.9 = dense healthy crop), from red and near-infrared light.
- **ESI (Evaporative Stress Index):** ECOSTRESS ratio of actual to potential ET. Lower = plants are closing their
  pores to save water, i.e. water stress.
- **Frost risk:** chance of a night below 0 C after a given date. "1 in 10" = in 1 of 10 years it is even later.
- **Hot day:** maximum temperature above 30 C. Heat at flowering damages grain set.
- **Spatial cross-validation / R2:** the model is always tested on areas it never saw. R2 = share of the
  variation explained (0 = useless, 1 = perfect).
""")
    with st.expander("Satellite vs ground measurements", expanded=False):
        st.markdown("""
Satellites see **every field, every year, everywhere**, but measure indirectly (light, microwaves, gravity) and
at coarser scales; clouds create gaps. **Ground stations** (e.g. German Weather Service) measure directly and
precisely, but only at a few points. Combining both is best: ground data check and correct the satellites,
satellites fill the space between stations. FARMS uses satellite and model data throughout and states their
resolution; checking them against a nearby weather station is part of our validation.
""")
        dpath = DATA_PROCESSED / "dwd_check.json"
        if dpath.exists():
            dw = json.loads(dpath.read_text())
            st.markdown(f"**Our check:** NASA POWER at {dw['compared_field'].replace('_', ' ')} vs the German "
                        f"Weather Service station **{dw['station_name']}** ({dw['distance_km']} km away), "
                        f"{dw['period']}, {dw['n_days']:,} days.")
            q = st.columns(4)
            q[0].metric("Temperature match (monthly r)", dw["temp_r"], f"{dw['temp_bias_c']:+.1f} C bias",
                        delta_color="off")
            q[1].metric("Rain match (monthly r)", dw["rain_r"], f"{dw['rain_bias_pct']:+.0f} % vs station",
                        delta_color="off")
            q[2].metric("Hot days/yr: POWER vs station", f"{dw['hot_days_power_yr']} vs {dw['hot_days_station_yr']}")
            q[3].metric("Heavy rain days/yr: POWER vs station",
                        f"{dw['heavy_rain_power_yr']} vs {dw['heavy_rain_station_yr']}")
            mpath = DATA_PROCESSED / "dwd_monthly.csv"
            if mpath.exists():
                mo = pd.read_csv(mpath)
                mo["date"] = pd.to_datetime(mo["month"])
                mo = mo[mo["date"].dt.year >= mo["date"].dt.year.max() - 9]
                fdw = go.Figure()
                fdw.add_trace(go.Bar(x=mo["date"], y=mo["rain_station"], name="Rain, station (mm)",
                                     marker_color="#185FA5", opacity=0.55))
                fdw.add_trace(go.Bar(x=mo["date"], y=mo["rain_power"], name="Rain, NASA POWER (mm)",
                                     marker_color="#97C459", opacity=0.75))
                fdw.add_trace(go.Scatter(x=mo["date"], y=mo["t_station"], name="Temp, station (C)", yaxis="y2",
                                         line=dict(color="#A32D2D")))
                fdw.add_trace(go.Scatter(x=mo["date"], y=mo["t_power"], name="Temp, NASA POWER (C)", yaxis="y2",
                                         line=dict(color="#EF9F27", dash="dot")))
                fdw.update_layout(height=360, barmode="overlay", margin=dict(t=30, b=10),
                                  title="Monthly values, last 10 years",
                                  yaxis=dict(title="rain (mm/month)"),
                                  yaxis2=dict(title="temperature (C)", overlaying="y", side="right", dtick=10, showgrid=False),
                                  legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fdw, **WIDE)
            t_ok = dw["temp_r"] >= 0.95 and abs(dw["temp_bias_c"]) <= 1.0
            r_ok = dw["rain_r"] >= 0.8 and abs(dw["rain_bias_pct"]) <= 15
            msg = ("Temperature from NASA POWER matches the station well, so frost, heat and ET0 numbers can be "
                   "trusted. " if t_ok else
                   f"Temperature differs from the station by {dw['temp_bias_c']:+.1f} C on average: treat frost, heat "
                   "and ET0 numbers with care. ")
            msg += ("Rain also matches month to month. " if r_ok else
                    "Rain matches less closely: a ~50 km grid cell cannot see every local shower, so single-season "
                    "rain is one of the least certain inputs. ")
            if dw["heavy_rain_power_yr"] < 0.7 * dw["heavy_rain_station_yr"]:
                msg += ("But a ~50 km grid smooths out downpours, so the app's heavy-rain counts are likely too low "
                        f"({dw['heavy_rain_power_yr']} vs {dw['heavy_rain_station_yr']} days/yr at the station). ")
            if abs(dw["hot_days_power_yr"] - dw["hot_days_station_yr"]) > 3:
                msg += (f"Hot-day counts differ ({dw['hot_days_power_yr']} vs {dw['hot_days_station_yr']} per year): "
                        "read them as a trend, not an exact number. ")
            st.caption("What this means: " + msg + "Source: " + dw.get("source", "DWD") + ".")
        else:
            st.caption("Station check not run yet (python src/dwd_check.py).")
    with st.expander("How different crop types help soil health", expanded=False):
        st.markdown("""
- **Nitrogen-fixing legumes** (lupin, pea, soybean, lucerne, clover): bacteria in root nodules take nitrogen from
  the air, so the next crop needs less fertiliser. Lupin is adapted to acid sandy soils.
- **Cover crops** (phacelia, vetch-rye): grow in gaps between main crops, protect bare soil from wind and heavy
  rain, catch leftover nutrients before they wash out, and feed soil life with roots.
- **Deep-rooting crops** (lucerne, sunflower, sorghum, rye): reach water and nutrients in the subsoil and leave
  channels that help later crops and rain soak in (infiltration).
- **Drought-tolerant crops** (rye, sorghum, millet, sunflower, lupin): keep producing biomass in dry years, so
  more roots and residues return carbon to the soil instead of failing.
- **Diverse rotations:** different crop families break pest and disease cycles (e.g. rye after rye builds up
  diseases), and alternating root depths uses the whole soil profile.
""")

    st.subheader("Data used")
    data = pd.DataFrame([
        ["NASA POWER", "Daily temperature, rain, humidity, wind, radiation", "~50 km", "1981-2026", "Frost, heat, ET0, water balance"],
        ["NASA SMAP L4", "Root-zone soil moisture", "9 km", "2015-2026", "Soil moisture trend, ML feature"],
        ["NASA GPM IMERG", "Monthly rain", "~11 km", "2000-2025", "ML feature, rain reliability"],
        ["NASA MODIS", "NDVI, land surface temperature, ET/PET", "250 m-1 km", "2000-2026", "ML target, heat, water use"],
        ["NASA ECOSTRESS", "Evaporative stress, ET, LST, soil moisture", "70 m", "2018-2026", "Field-level stress check"],
        ["NASA GLDAS", "3-hourly air temperature", "~25 km", "2000-2026", "Heat days, frost, GDD for ML"],
        ["NASA GRACE-FO", "Total water storage", "~300 km", "2002-2024", "Regional groundwater context"],
        ["NASA SRTM", "Elevation, slope", "30 m", "static", "Erosion, terrain features"],
        ["ESA Sentinel-2", "NDVI, scene classes", "10 m", "2017-2026", "Bare-soil months"],
        ["ISRIC SoilGrids", "Texture, carbon, pH, density", "250 m", "static", "Soil water store, crop fit"],
        ["JRC EUCROPMAP", "Crop type", "10 m", "2018, 2022", "ML labels, current rotation"],
        ["FAO-56", "Crop coefficients, root depths", "-", "-", "Crop water demand"],
    ], columns=["Source", "What", "Resolution", "Years", "Used for"])
    st.dataframe(data, hide_index=True, **WIDE)

    st.subheader("Limitations (read before acting)")
    qc = D["qc"].get(fid) or {}
    _dp = DATA_PROCESSED / "dwd_check.json"
    _dw = json.loads(_dp.read_text()) if _dp.exists() else None
    station_line = (f"- **Heavy rain is undercounted:** against the Cottbus weather station, NASA POWER shows "
                    f"{_dw['heavy_rain_power_yr']} vs {_dw['heavy_rain_station_yr']} days/yr with >= 20 mm "
                    f"(temperature r = {_dw['temp_r']}, monthly rain r = {_dw['rain_r']}). Erosion risk from "
                    "downpours is likely higher than the app shows.\n" if _dw else "")
    st.markdown(f"""
- **Supports, does not replace,** farmer and adviser judgement. Every suggestion lists what to check locally.
- **Only 2 years of crop labels** (2018, 2022). Weather variety comes from many places, not many years; year-to-year accuracy is moderate.
- **NDVI is not yield.** It measures greenness/vigour, a common proxy.
- **Coarse weather data:** NASA POWER cells are ~50 km, so two of the demo fields share one cell; field differences come from soil, slope, satellite stress and crops.
- **Soil values are modelled** (SoilGrids 250 m). Use the soil-test box in the sidebar if you have lab results.
- **ECOSTRESS** {f"readings are irregular and cloud-affected: for this field {qc['share_cloudy_pct']} % of overpasses were removed as cloudy; 2018-2019 marked low confidence." if qc.get('share_cloudy_pct') is not None else "data was not downloaded for this field (optional); nothing else depends on it."}
- **Prices, yields and costs are indicative**, not market data. Crop-table tolerances are expert judgement (sources in data/crops.json).
- **Few training labels** for sorghum, millet, lupin and forage crops: their reliability uses the more cautious of model and crop table.
- **Earth Engine datasets end on different dates** (IMERG Sept 2025, GRACE Sept 2024).
""" + station_line)