# Electricity Consumption Forecasting & Anomaly Detection

**Live app:** https://electricity-forecasting-dszsxnebf7qghzho5p9anp.streamlit.app/

A machine learning project that forecasts hourly household electricity consumption, flags unusual usage, and presents the results in a deployed web app. Built as the capstone project for the TechCrush AI Bootcamp.

## Problem Statement

Households and energy providers benefit from knowing how much electricity will be used in the coming hours, and from being alerted when usage deviates sharply from what is expected. This project builds a forecasting pipeline that predicts hourly electricity consumption from past usage patterns, compares several models against a simple baseline, and flags hours where actual consumption differs sharply from the forecast.

## Dataset

**UCI Individual Household Electric Power Consumption**: about 2 million minute-level measurements from one household in Sceaux, France (December 2006 to November 2010).

- **Target variable:** `Global_active_power` (kilowatts)
- **Other variables:** reactive power, voltage, current intensity, and three sub-meters (kitchen, laundry room, water heater and air conditioner)
- **License:** CC BY 4.0

> Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

## Methodology

1. **Data cleaning.** Missing readings (marked "?", about 1.25% of rows) were identified. Date and time columns were combined into a single timestamp, and the data was checked for duplicates, negative values and out-of-range values.
2. **Resampling.** Minute-level readings were averaged to hourly values (about 34,000 hours). Short gaps of up to 6 hours were filled by interpolation; longer outages were left missing and excluded from modeling.
3. **Feature engineering.** Each hour was described using only information available before it: hour of day, day of week, month, consumption 1 hour ago, 24 hours ago and 168 hours ago (one week), and the average of the previous 24 hours.
4. **Time-based split.** Models were trained on data before 1 June 2010 and tested on data from June 2010 onward. Splitting by time, rather than randomly, prevents the model from seeing the future during training.
5. **Models compared.**
   - **Baseline:** predict the value from the same hour the previous day
   - **Linear Regression**
   - **Random Forest** (100 trees, max depth 12)
6. **Evaluation.** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).
7. **Anomaly flagging.** An hour is flagged when the difference between actual and forecast consumption exceeds 3 standard deviations of the forecast errors (adjustable in the app).

## Results

Evaluated on the held-out test period (June to November 2010):

| Model | MAE (kW) | RMSE (kW) | MAPE (%) |
|---|---|---|---|
| Baseline (yesterday) | 0.4847 | 0.7305 | 64.63 |
| Linear Regression | 0.3545 | 0.4968 | 51.17 |
| **Random Forest** | **0.3195** | **0.4619** | **44.94** |

**Key findings**

- Random Forest performed best on every metric. It reduced MAE by about 34% and RMSE by about 37% compared with the baseline.
- Linear Regression also improved on the baseline (about 27% lower MAE), showing that recent-usage features carry most of the useful signal.
- MAPE is high for all models because it divides by the actual value, so hours with very low consumption inflate the percentage. MAE and RMSE, which are in kilowatts, are the more reliable measures for this household.

## The App

The Streamlit app lets users:

- Choose a date range within the test period
- Compare actual and forecast consumption on a line chart
- See MAE and RMSE for the selected range
- Adjust the anomaly sensitivity and view the flagged hours
- Review the model comparison table

The app reads its results from the `results/` folder in this repository.

## Repository Structure

```
├── app/            Streamlit app (app.py) and its requirements
├── notebooks/      Cleaning, EDA, anomaly detection, clustering and modeling notebooks
├── processed/      clean_hourly.csv (cleaned, hourly data)
├── results/        predictions.csv and model_comparison.csv (read by the app)
├── models/         Saved models
├── src/            Reusable code
└── requirements.txt
```

## How to Run

**View the app online:** open the live link at the top of this page.

**Run the app locally:**

```
git clone https://github.com/dunurejoice20/electricity-forecasting.
cd electricity-forecasting.
pip install streamlit pandas numpy
streamlit run app/app.py
```

The app loads its results from this repository, so an internet connection is required.

**Reproduce the data and results:** open the notebooks in `notebooks/` in Google Colab. The raw dataset is downloaded from the UCI repository inside the notebook and is not stored in this repository because of its size.

## Limitations and Future Work

**Limitations**

- The data comes from a single household in France (2006 to 2010), so the results may not transfer to other households, climates or regions, including Nigerian or Ghanaian homes.
- The test period covers about six months, so seasonal effects are only partly represented.
- Models use the previous hour's consumption, so this is short-term (next-hour) forecasting. The baseline uses the same hour on the previous day, which is a simpler reference than what the models see.
- Some long outages in the data could not be filled and were excluded.
- Flagged anomalies are hours that are unusual compared with the forecast. They are not confirmed faults.

**Future work**

- Add a neural network (for example an LSTM) and compare it fairly using the same time-based split
- Tune model hyperparameters and add holiday and temperature features
- Forecast further ahead (for example 24 hours) instead of only the next hour
- Combine the forecast with Isolation Forest anomaly detection and daily usage-profile clustering explored in the notebooks
- Containerize the app with Docker for portable deployment
- Test on local (Nigerian or Ghanaian) household data

## Team

| Name | Role |
|---|---|
| Miracle Emeka | Data cleaning and EDA |
| Jubril Fahd | Modeling |
| Solomon Ojo | Anomaly detection and clustering |
| Rejoice Dunu | App, deployment and documentation |
