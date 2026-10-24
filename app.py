"""
Streamlit app: next-hour AQI forecast for Delhi (CPCB stations).

Run locally:   streamlit run app.py
Needs the files in ./artifacts (created by the notebook cell "STEP 44").
"""
import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from aqi_pipeline import (MIN_HISTORY_HOURS, RAW_COLUMNS, aqi_category, build_features,
                          check_continuity, latest_per_station, predict)

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Delhi next-hour AQI forecast", page_icon="🌫️", layout="wide")

# Results reported in the paper (test period 20 Jul – 1 Oct 2019, 10,467 station-hours).
PAPER_TABLE = pd.DataFrame(
    [["Persistence (AQI at t)", 0.9897, 1.886, 4.349, "reference"],
     ["Random Forest (main model)", 0.9936, 1.645, 3.427, "best point estimate"],
     ["Hybrid RF + XGBoost", 0.9933, 1.719, 3.500, "no gain over single models"],
     ["LightGBM", 0.9933, 1.817, 3.506, "RMSE indistinguishable from Random Forest"],
     ["XGBoost", 0.9923, 1.986, 3.755, "significantly worse than Random Forest"]],
    columns=["Model", "Test R²", "MAE", "RMSE", "Note"])


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load(ART / "model_lgbm.joblib")
    cols = json.loads((ART / "feature_cols.json").read_text())
    mapping = json.loads((ART / "station_mapping.json").read_text())
    metrics = json.loads((ART / "metrics.json").read_text())
    return model, cols, mapping, metrics


@st.cache_data(show_spinner="Preparing historical data…")
def load_history_predictions(_model, cols, mapping):
    hist = pd.read_csv(ART / "history_clean.csv.gz", parse_dates=["timestamp"])
    feats = build_features(hist, mapping, with_target=True)
    feats["forecast"] = predict(_model, feats, cols)
    return hist, feats


def category_badge(aqi):
    name, colour = aqi_category(aqi)
    return f"<span style='background:{colour};color:white;padding:2px 10px;border-radius:10px;font-weight:600'>{name}</span>"


def risk_warnings(row_time, aqi_now, unseen):
    msgs = []
    if row_time.month in (10, 11, 12, 1, 2):
        msgs.append("It is the post-monsoon / winter season. The model was evaluated on late July – September 2019, "
                    "and in forward-chaining validation every model dropped to R² ≈ 0.8 during the October–December rise. "
                    "Treat winter forecasts with caution.")
    if aqi_now > 300:
        msgs.append("The current AQI is above 300 (Very Poor or Severe), a range that is rare in the evaluation period.")
    if unseen:
        msgs.append("This station was not in the training data. Leave-one-station-out tests showed skill that did not "
                    "transfer consistently to unseen stations.")
    return msgs


def forecast_chart(window, forecast_time, aqi_now, pred, actual=None):
    base = alt.Chart(window).mark_line(point=True, color="#1F3864").encode(
        x=alt.X("timestamp:T", title="Time"), y=alt.Y("AQI:Q", title="AQI", scale=alt.Scale(zero=False)),
        tooltip=["timestamp:T", "AQI:Q"])
    pts = [{"timestamp": forecast_time, "AQI": pred, "series": "Model forecast"},
           {"timestamp": forecast_time, "AQI": aqi_now, "series": "Persistence (current AQI)"}]
    if actual is not None and not np.isnan(actual):
        pts.append({"timestamp": forecast_time, "AQI": actual, "series": "Actual"})
    points = alt.Chart(pd.DataFrame(pts)).mark_point(size=130, filled=True).encode(
        x="timestamp:T", y="AQI:Q",
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Model forecast", "Persistence (current AQI)", "Actual"],
            range=["#E76F51", "#7F7F7F", "#2E7D6B"]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["series:N", "AQI:Q"])
    return (base + points).properties(height=320)


# ----------------------------------------------------------------------------- app
st.title("🌫️ Delhi next-hour AQI forecast")
st.caption("Forecasts the CPCB Air Quality Index one hour ahead from the previous 72 h of station data. "
           "Model: LightGBM on 159 engineered features, trained on six Delhi stations (June 2018 – July 2019).")

if not (ART / "model_lgbm.joblib").exists():
    st.error("Model files not found in the `artifacts` folder. Run the notebook cell 'STEP 44' and copy the "
             "exported files into `artifacts/` (see README.md).")
    st.stop()

model, FEATURE_COLS, STATION_MAP, METRICS = load_artifacts()
hist, feats = load_history_predictions(model, FEATURE_COLS, STATION_MAP)
dep = METRICS["deployed_test"]

with st.sidebar:
    st.header("About this forecast")
    st.markdown(
        f"**Typical error (test period):** MAE {dep['MAE']:.2f} AQI points, RMSE {dep['RMSE']:.2f}.\n\n"
        f"**90% of errors** were below {METRICS['abs_error_q90']:.1f} AQI points.\n\n"
        "The test period (20 Jul – 1 Oct 2019) was a low-pollution season (mean AQI "
        f"{METRICS['test_period']['aqi_mean']:.0f}), so errors in winter are likely larger.")
    st.warning("Research prototype. Not for health or regulatory decisions.")

tab_fc, tab_ev, tab_about = st.tabs(["Forecast", "How good is it?", "Data format & limits"])

# ----------------------------------------------------------------------------- forecast tab
with tab_fc:
    mode = st.radio("Data source", ["Replay historical data", "Upload recent data"], horizontal=True)

    if mode == "Replay historical data":
        st.write("Pick a station and an hour. The app forecasts the next hour and, because this is historical data, "
                 "shows what actually happened.")
        c1, c2 = st.columns([1, 3])
        station = c1.selectbox("Station", sorted(feats["Station"].unique()))
        sf = feats[feats["Station"] == station].sort_values("timestamp").reset_index(drop=True)
        t_min, t_max = sf["timestamp"].min().to_pydatetime(), sf["timestamp"].max().to_pydatetime()
        default = pd.Timestamp(METRICS["test_period"]["start"]).to_pydatetime()
        default = min(max(default, t_min), t_max)
        t_sel = c2.slider("Forecast made at", min_value=t_min, max_value=t_max, value=default,
                          step=timedelta(hours=1), format="YYYY-MM-DD HH:mm")
        idx = (sf["timestamp"] - pd.Timestamp(t_sel)).abs().idxmin()
        row = sf.loc[idx]
        window = sf.loc[max(0, idx - 47):idx, ["timestamp", "AQI"]]
        t_next = row["timestamp"] + pd.Timedelta(hours=1)
        pred = float(row["forecast"])
        actual = float(row["AQI_next_1h"])
        aqi_now = float(row["AQI"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current AQI", f"{aqi_now:.0f}")
        m2.metric("Forecast (next hour)", f"{pred:.0f}", f"{pred - aqi_now:+.0f} vs now")
        m3.metric("Actual (next hour)", f"{actual:.0f}")
        m4.metric("Model error", f"{pred - actual:+.1f}", f"persistence error {aqi_now - actual:+.1f}", delta_color="off")
        st.markdown(f"Now: {category_badge(aqi_now)} &nbsp;→&nbsp; forecast: {category_badge(pred)}", unsafe_allow_html=True)
        for msg in risk_warnings(row["timestamp"], aqi_now, unseen=False):
            st.warning(msg)
        st.altair_chart(forecast_chart(window, t_next, aqi_now, pred, actual), width="stretch")

    else:
        st.write(f"Upload hourly station data with at least {MIN_HISTORY_HOURS} consecutive rows per station "
                 "(more is fine; the most recent rows are used).")
        example = hist.sort_values(["Station", "timestamp"]).groupby("Station").tail(72)
        st.download_button("Download an example CSV (last 72 h of history)", example.to_csv(index=False),
                           file_name="example_input.csv", mime="text/csv")
        up = st.file_uploader("CSV file", type="csv")
        use_example = st.checkbox("Use the example data instead of uploading")
        raw = None
        if up is not None:
            raw = pd.read_csv(up)
        elif use_example:
            raw = example.copy()
        if raw is not None:
            problems = check_continuity(raw)
            if problems:
                st.error("The data cannot be used yet:")
                for p in problems:
                    st.write("• " + p)
            else:
                f = build_features(raw, STATION_MAP, with_target=False)
                last = latest_per_station(f)
                if last.empty:
                    st.error("No station has enough clean history to build the features.")
                else:
                    last["forecast"] = predict(model, last, FEATURE_COLS)
                    last["forecast_for"] = last["timestamp"] + pd.Timedelta(hours=1)
                    unseen = {s: s not in STATION_MAP for s in last["Station"]}
                    table = pd.DataFrame({
                        "Station": last["Station"], "Last observed hour": last["timestamp"],
                        "Current AQI": last["AQI"].round(0), "Forecast for": last["forecast_for"],
                        "Forecast AQI": last["forecast"].round(0),
                        "Change": (last["forecast"] - last["AQI"]).round(0) + 0.0,
                        "Category now": [aqi_category(a)[0] for a in last["AQI"]],
                        "Category forecast": [aqi_category(a)[0] for a in last["forecast"]]})
                    st.dataframe(table, hide_index=True)
                    pick = st.selectbox("Show station", list(last["Station"]))
                    r = last[last["Station"] == pick].iloc[0]
                    st.markdown(f"**{pick}:** now {category_badge(r['AQI'])} → forecast {category_badge(r['forecast'])}",
                                unsafe_allow_html=True)
                    for msg in risk_warnings(r["timestamp"], r["AQI"], unseen[pick]):
                        st.warning(msg)
                    w = f[f["Station"] == pick].tail(48)[["timestamp", "AQI"]]
                    st.altair_chart(forecast_chart(w, r["forecast_for"], float(r["AQI"]), float(r["forecast"])),
                                    width="stretch")
                    st.caption(f"Typical error: ±{dep['MAE']:.1f} AQI points (MAE); 90% of test-period errors were below "
                               f"{METRICS['abs_error_q90']:.1f}. These come from a low-pollution season.")

# ----------------------------------------------------------------------------- evidence tab
with tab_ev:
    st.subheader("Deployed model (LightGBM) on the held-out test period")
    skill_rmse = 1 - dep["RMSE"] / METRICS["persistence_test"]["RMSE"]
    a, b, c, d = st.columns(4)
    a.metric("R²", f"{dep['R2']:.4f}", f"{dep['R2'] - METRICS['persistence_test']['R2']:+.4f} vs persistence")
    b.metric("MAE", f"{dep['MAE']:.3f}")
    c.metric("RMSE", f"{dep['RMSE']:.3f}")
    d.metric("RMSE skill vs persistence", f"{100 * skill_rmse:.1f}%")
    st.subheader("Models compared in the paper")
    st.dataframe(PAPER_TABLE, hide_index=True)
    st.markdown(
        "**How to read this.**\n"
        "- A naive *persistence* forecast (next hour = current AQI) already reaches R² = 0.9897. "
        "The best model improves R² by only 0.0039, a 21% reduction in RMSE.\n"
        "- Random Forest, LightGBM and the hybrid are statistically indistinguishable in RMSE, so the app uses LightGBM, "
        "which trains in seconds and gives a small model file.\n"
        "- The improvement over persistence comes mainly from reducing large errors, not from being better in most hours "
        "(the model has a lower absolute error in 51% of hours, which cannot be told apart from 50%).\n"
        "- Models trained before the high-pollution season performed much worse when it began (R² ≈ 0.8 for mid-October to "
        "late December 2018), and the skill did not transfer to every unseen station.")

# ----------------------------------------------------------------------------- about tab
with tab_about:
    st.subheader("Input format")
    st.write(f"One row per station and hour, at least {MIN_HISTORY_HOURS} consecutive hours per station, "
             "with these columns:")
    st.code(", ".join(RAW_COLUMNS))
    st.markdown(
        "- `timestamp` is the hour of observation (e.g. `2019-07-20 15:00`); `Station` is the station name.\n"
        "- Pollutants are in CPCB units (µg/m³; CO in mg/m³). Meteorology: AT °C, BP mmHg, SR W/m², RH %, WS m/s, WD °.\n"
        "- Rows must be consecutive hours without gaps or missing values: lags and rolling windows are row-based, "
        "as in training.\n"
        f"- Known stations: {', '.join(sorted(STATION_MAP))}.")
    st.subheader("What the model does")
    st.markdown(
        "The pipeline recomputes CPCB sub-indices from 24 h (and 8 h) pollutant averages, the current AQI as their maximum, "
        "pollutant and AQI lags (1–24 h), rolling means and standard deviations, calendar encodings and a station code. "
        "The target is the AQI one hour ahead.")
    st.subheader("Limitations")
    st.markdown(
        "- Trained on one city, six stations and about 13 months; evaluated on late July – September 2019 only.\n"
        "- Skill over simply repeating the current AQI is modest, and it fell sharply in the post-monsoon rise.\n"
        "- Not validated for stations outside the six used in training, or for later years.\n"
        "- Research prototype, not a regulatory or health advisory tool.")
