"""
Electricity Consumption Forecasting & Anomaly Detection
Starter app (PLACEHOLDER DATA). Replace the marked sections with the real
data, model and anomaly logic once Persons 1-3 have shared their outputs.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Electricity Forecasting", layout="wide")


# ---------------------------------------------------------------
# 1. DATA  (PLACEHOLDER)
# Later: replace with the real cleaned file from Person 1, e.g.
#   hourly = pd.read_csv("processed/clean_hourly.csv",
#                        parse_dates=["datetime"], index_col="datetime")
# ---------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-01", "2010-03-31 23:00", freq="h")
    hour = idx.hour.to_numpy()
    daily = 1.0 + 0.8 * np.sin((hour - 6) / 24 * 2 * np.pi)   # daily cycle
    weekend = np.where(idx.dayofweek >= 5, 0.3, 0.0)           # weekend bump
    actual = daily + weekend + rng.normal(0, 0.15, len(idx))
    actual = np.clip(actual, 0.1, None)
    df = pd.DataFrame({"actual": actual}, index=idx)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------
# 2. FORECAST  (PLACEHOLDER)
# Later: load Person 2's saved model and predict, e.g.
#   model = joblib.load("models/best_model.joblib")
#   df["forecast"] = model.predict(features)
# ---------------------------------------------------------------
def add_forecast(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    out = df.copy()
    out["forecast"] = out["actual"].shift(24).bfill() * 0.9 + rng.normal(0, 0.05, len(out))
    return out


# ---------------------------------------------------------------
# 3. ANOMALIES  (PLACEHOLDER)
# Later: use Person 3's method (Isolation Forest or residual threshold).
# ---------------------------------------------------------------
def flag_anomalies(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    out["residual"] = out["actual"] - out["forecast"]
    threshold = n_std * out["residual"].std()
    out["is_anomaly"] = out["residual"].abs() > threshold
    return out


# ---------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------
st.title("Electricity Consumption Forecasting & Anomaly Detection")
st.caption(
    "Prototype using placeholder data. Real UCI household data and trained "
    "model will replace it."
)

data = flag_anomalies(add_forecast(load_data()))

# Sidebar controls
st.sidebar.header("Controls")
min_d, max_d = data.index.min().date(), data.index.max().date()
picked = st.sidebar.date_input(
    "Date range",
    value=(min_d, min_d + pd.Timedelta(days=14)),
    min_value=min_d,
    max_value=max_d,
)
if not isinstance(picked, (tuple, list)) or len(picked) < 2:
    st.info("Pick both a start and an end date in the sidebar.")
    st.stop()
start, end = picked
n_std = st.sidebar.slider("Anomaly sensitivity (std devs)", 2.0, 5.0, 3.0, 0.5)

view = flag_anomalies(data[["actual", "forecast"]], n_std)
view = view.loc[str(start): str(end)]

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
anoms = view[view["is_anomaly"]][["actual", "forecast", "residual"]]
if anoms.empty:
    st.info("No anomalies in this range at the current sensitivity.")
else:
    st.dataframe(anoms.round(3), use_container_width=True)

# Model comparison (PLACEHOLDER numbers, replace with Person 2's results)
st.subheader("Model comparison")
st.dataframe(
    pd.DataFrame(
        {
            "Model": ["Baseline (yesterday)", "Linear Regression", "Random Forest", "Neural Network"],
            "MAE": [None, None, None, None],
            "RMSE": [None, None, None, None],
            "MAPE (%)": [None, None, None, None],
        }
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("Results will be filled in once modeling is complete.")
