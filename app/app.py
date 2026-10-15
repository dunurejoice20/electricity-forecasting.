"""
Electricity Consumption Forecasting & Anomaly Detection
Reads real results (predictions + model comparison) from the GitHub repo.
Falls back to placeholder data if the files can't be loaded.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Electricity Forecasting", layout="wide")

# ---------------------------------------------------------------
# CONFIG: change only this if the repo name or branch changes
# ---------------------------------------------------------------
BASE = "https://raw.githubusercontent.com/dunurejoice20/electricity-forecasting./main"
PRED_URL = f"{BASE}/results/predictions.csv"
COMP_URL = f"{BASE}/results/model_comparison.csv"


# ---------------------------------------------------------------
# DATA
# ---------------------------------------------------------------
def placeholder_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-01", "2010-03-31 23:00", freq="h")
    hour = idx.hour.to_numpy()
    daily = 1.0 + 0.8 * np.sin((hour - 6) / 24 * 2 * np.pi)
    actual = np.clip(daily + rng.normal(0, 0.15, len(idx)), 0.1, None)
    df = pd.DataFrame({"actual": actual}, index=idx)
    df["forecast"] = df["actual"].shift(24).bfill() * 0.9
    df.index.name = "datetime"
    return df


@st.cache_data(ttl=600)
def load_data():
    """Returns (dataframe, using_real_data)."""
    try:
        df = pd.read_csv(PRED_URL, parse_dates=["datetime"], index_col="datetime")
        return df[["actual", "forecast"]].dropna().sort_index(), True
    except Exception:
        return placeholder_data(), False


@st.cache_data(ttl=600)
def load_comparison():
    try:
        return pd.read_csv(COMP_URL)
    except Exception:
        return None


def flag_anomalies(df: pd.DataFrame, n_std: float) -> pd.DataFrame:
    out = df.copy()
    out["residual"] = out["actual"] - out["forecast"]
    threshold = n_std * out["residual"].std()
    out["is_anomaly"] = out["residual"].abs() > threshold
    return out


# ---------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------
st.title("Electricity Consumption Forecasting & Anomaly Detection")

data, is_real = load_data()

if is_real:
    st.caption(
        "UCI Individual Household Electric Power Consumption data (hourly). "
        "Forecasts are on a held-out test period the model never saw in training."
    )
else:
    st.warning("Could not load the results files, showing placeholder data instead.")

# Sidebar controls
st.sidebar.header("Controls")
min_d, max_d = data.index.min().date(), data.index.max().date()
default_start = max(min_d, (data.index.max() - pd.Timedelta(days=14)).date())

picked = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_d),
    min_value=min_d,
    max_value=max_d,
)
if not isinstance(picked, (tuple, list)) or len(picked) < 2:
    st.info("Pick both a start and an end date in the sidebar.")
    st.stop()
start, end = picked

n_std = st.sidebar.slider("Anomaly sensitivity (std devs)", 2.0, 5.0, 3.0, 0.5)
st.sidebar.caption("Higher value = fewer, more extreme anomalies flagged.")

# Anomalies are judged against the whole period, then filtered to the view
full = flag_anomalies(data, n_std)
view = full.loc[str(start): str(end)]

if view.empty:
    st.info("No data in this range. Try a different one.")
    st.stop()

# Key numbers
mae = view["residual"].abs().mean()
rmse = np.sqrt((view["residual"] ** 2).mean())
c1, c2, c3 = st.columns(3)
c1.metric("MAE (kW)", f"{mae:.3f}")
c2.metric("RMSE (kW)", f"{rmse:.3f}")
c3.metric("Anomalies flagged", int(view["is_anomaly"].sum()))

# Forecast chart
st.subheader("Actual vs. forecast")
st.line_chart(view[["actual", "forecast"]])

# Anomalies
st.subheader("Flagged anomalies")
st.caption("Hours where actual usage differs sharply from the forecast.")
anoms = view[view["is_anomaly"]][["actual", "forecast", "residual"]]
if anoms.empty:
    st.info("No anomalies in this range at the current sensitivity.")
else:
    st.dataframe(anoms.round(3), use_container_width=True)

# Model comparison
st.subheader("Model comparison")
comparison = load_comparison()
if comparison is not None:
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("Evaluated on a time-based split: trained on the past, tested on later data.")
else:
    st.info("Model comparison results are not available yet.")
