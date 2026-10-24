"""
Feature pipeline for next-hour AQI forecasting.

A faithful port of the notebook steps that build the 159 model features:
  Step 14  24 h / 8 h rolling averages
  Step 19  CPCB sub-indices (concentration rounded before breakpoint lookup) and AQI = max sub-index
  Step 17  keep rows whose PM2.5, PM10, NO2 and SO2 24 h averages exist
  Step 21  calendar encodings and station code
  Step 26  pollutant lags, rolling statistics and changes
  Step 31  AQI lags and rolling statistics

Lags and rolling windows are ROW-based inside each station, exactly as in the notebook,
so the input must be consecutive hourly rows per station (use `check_continuity`).
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["Station", "timestamp", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2",
               "CO", "Ozone", "Benzene", "AT", "BP", "SR", "RH", "WS", "WD"]
MIN_HISTORY_HOURS = 72          # consecutive hourly rows per station needed for one forecast (averaging window + chained lags)

AQI_BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300),
              (121, 250, 301, 400), (251, float("inf"), 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300),
             (351, 430, 301, 400), (431, float("inf"), 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300),
            (281, 400, 301, 400), (401, float("inf"), 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300),
            (801, 1600, 301, 400), (1601, float("inf"), 401, 500)],
    "Ozone": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300),
              (209, 748, 301, 400), (749, float("inf"), 401, 500)],
    "NH3": [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300),
            (1201, 1800, 301, 400), (1801, float("inf"), 401, 500)],
    "CO": [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200), (10.1, 17.0, 201, 300),
           (17.1, 34.0, 301, 400), (34.1, float("inf"), 401, 500)],
}

AQI_CATEGORIES = [(50, "Good", "#00B050"), (100, "Satisfactory", "#92D050"), (200, "Moderate", "#FFC000"),
                  (300, "Poor", "#ED7D31"), (400, "Very Poor", "#C00000"), (10 ** 9, "Severe", "#7030A0")]


def aqi_category(aqi):
    if pd.isna(aqi):
        return "Missing", "#7F7F7F"
    for upper, name, colour in AQI_CATEGORIES:
        if aqi <= upper:
            return name, colour


def _subindex_value(conc, pollutant):
    c = round(float(conc))
    for c_low, c_high, i_low, i_high in AQI_BREAKPOINTS[pollutant]:
        if c_low <= c <= c_high:
            if c_high == c_low:
                return float(i_low)
            return ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
    return np.nan


def subindex(series, pollutant):
    """Corrected sub-index (Step 19). Evaluated once per distinct rounded concentration."""
    rounded = series.round()
    table = {v: _subindex_value(v, pollutant) for v in rounded.dropna().unique()}
    return rounded.map(table)


def check_continuity(raw, min_rows=MIN_HISTORY_HOURS):
    """Return a list of human-readable problems with the input (empty list = OK)."""
    problems = []
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    t = pd.to_datetime(raw["timestamp"], errors="coerce")
    if t.isna().any():
        problems.append("Some timestamps could not be parsed.")
    d = raw.assign(timestamp=t).sort_values(["Station", "timestamp"])
    for st, g in d.groupby("Station"):
        if len(g) < min_rows:
            problems.append(f"{st}: only {len(g)} rows, at least {min_rows} consecutive hourly rows are needed.")
            continue
        tail = g.tail(min_rows)
        gaps = tail["timestamp"].diff().dropna()
        if (gaps != pd.Timedelta(hours=1)).any():
            problems.append(f"{st}: the last {min_rows} rows are not consecutive hourly observations.")
        nan_cols = [c for c in RAW_COLUMNS[2:] if tail[c].isna().any()]
        if nan_cols:
            problems.append(f"{st}: missing values in the last {min_rows} rows for {', '.join(nan_cols)}.")
    return problems


def build_features(raw, station_mapping, with_target=False):
    """
    raw             DataFrame with RAW_COLUMNS (hourly, per station).
    station_mapping dict Station -> integer code used at training time.
    with_target     if True also create AQI_next_1h (used for replay and parity checks).
    Returns one row per station-hour for which every feature is available.
    """
    df = raw[RAW_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["Station", "timestamp"]).reset_index(drop=True)
    g = df.groupby("Station")

    # Step 14 – averaging windows
    for p in ["PM2.5", "PM10", "NO2", "SO2", "NH3"]:
        df[f"{p}_24h"] = g[p].transform(lambda x: x.rolling(24, min_periods=24).mean())
    for p in ["CO", "Ozone"]:
        df[f"{p}_8h"] = g[p].transform(lambda x: x.rolling(8, min_periods=8).mean())

    # Step 19 – sub-indices and AQI
    si_cols = []
    for p, src in [("PM2.5", "PM2.5_24h"), ("PM10", "PM10_24h"), ("NO2", "NO2_24h"), ("SO2", "SO2_24h"),
                   ("NH3", "NH3_24h"), ("CO", "CO_8h"), ("Ozone", "Ozone_8h")]:
        df[f"{p}_SI"] = subindex(df[src], p)
        si_cols.append(f"{p}_SI")
    df["AQI"] = df[si_cols].max(axis=1).round().clip(0, 500)

    # Step 17 – rows with the major 24 h averages available
    valid = df[["PM2.5_24h", "PM10_24h", "NO2_24h", "SO2_24h"]].notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)

    # Step 21 – calendar and station encodings
    ts = df["timestamp"]
    df["year"], df["month"], df["day"], df["hour"] = ts.dt.year, ts.dt.month, ts.dt.day, ts.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["Station_Code"] = df["Station"].map(station_mapping).fillna(-1).astype(int)

    # Step 26 – pollutant lags, rolling statistics, changes
    g = df.groupby("Station")
    new = {}
    for p in ["PM2.5", "PM10", "NO2", "NO", "SO2", "Ozone", "CO"]:
        for lag in [1, 3, 6, 12, 24]:
            new[f"{p}_lag_{lag}h"] = g[p].shift(lag)
        shifted = g[p].shift(1)
        for w in [3, 6, 12, 24]:
            new[f"{p}_rolling_mean_{w}h"] = shifted.groupby(df["Station"]).transform(lambda x: x.rolling(w).mean())
            new[f"{p}_rolling_std_{w}h"] = shifted.groupby(df["Station"]).transform(lambda x: x.rolling(w).std())
        new[f"{p}_change_1h"] = df[p] - new[f"{p}_lag_1h"]
        new[f"{p}_change_6h"] = df[p] - new[f"{p}_lag_6h"]
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    poll_feat = [c for c in df.columns if ("_lag_" in c or "_rolling_" in c or "_change_" in c)]
    df = df.dropna(subset=poll_feat).reset_index(drop=True)

    # Step 31 – AQI lags and rolling statistics
    g = df.groupby("Station")
    new = {}
    for lag in [1, 3, 6, 12, 24]:
        new[f"AQI_lag_{lag}h"] = g["AQI"].shift(lag)
    shifted = g["AQI"].shift(1)
    for w in [3, 6, 12, 24]:
        new[f"AQI_rolling_mean_{w}h"] = shifted.groupby(df["Station"]).transform(lambda x: x.rolling(w).mean())
        new[f"AQI_rolling_std_{w}h"] = shifted.groupby(df["Station"]).transform(lambda x: x.rolling(w).std())
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    aqi_feat = [c for c in df.columns if c.startswith("AQI_")]
    if with_target:
        df["AQI_next_1h"] = df.groupby("Station")["AQI"].shift(-1)
        df = df.dropna(subset=aqi_feat + ["AQI_next_1h"])
    else:
        df = df.dropna(subset=aqi_feat)
    return df.reset_index(drop=True)


def latest_per_station(feats):
    """Most recent feature row of each station (the hour to forecast from)."""
    return feats.sort_values("timestamp").groupby("Station").tail(1).sort_values("Station").reset_index(drop=True)


def predict(model, feats, feature_cols):
    X = feats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return model.predict(X)
