### Dataset Inspection Findings
The dataset inspection returned 2,034,118 rows and 16 columns. The columns include the original household electricity measurements, date and time information, and engineered features such as hour, day of the week, month, one-hour lag, 24-hour lag, and one-hour rolling mean.
The `datetime` column is stored in datetime format, while the electricity measurements and lag-based features are stored as floating-point values. The hour, day-of-week, and month features are stored as integer values.
The missing-value check returned zero missing values for all 16 columns in the inspected DataFrame. These findings provide an initial basis for the next stage of model development.

### Timestamp and Target Variable Analysis
The dataset is chronologically ordered and covers the period from 17 December 2006 to 26 November 2010. The timestamp interval analysis indicates that most consecutive observations are one minute apart. However, 133 intervals differ from one minute, indicating that gaps exist in the time series.
The descriptive statistics for `Global_active_power` show a mean of approximately 1.09, a standard deviation of approximately 1.06, a minimum of 0.076, and a maximum of 11.122.
Because the dataset contains gaps in its timestamps, row-based lag features require further validation. The feature-engineering process will therefore be revised to align readings to a continuous minute-by-minute timeline before creating the final lag and rolling-mean features.

### Data Preparation and Feature Engineering Results
The original DataFrame contained 2,034,118 observations. No duplicate timestamps were found during the preparation of the modeling dataset.
The data was sorted chronologically and indexed by timestamp. A continuous one-minute timeline was created to support time-based feature engineering. The target variable, `Global_active_power`, was retained alongside the hour, day of the week, and month features.
Additional features were generated using the previous minute's consumption, consumption 60 minutes earlier, consumption 1,440 minutes earlier, and the rolling mean of the preceding 60 minutes. Rows with missing values in the required target and feature columns were removed.
The resulting modeling dataset contained 2,015,236 rows and nine columns, with no missing values in the displayed columns. The prepared data will be used for chronological model training and evaluation.

### Training and Testing Strategy
The prepared dataset is divided chronologically into training and testing sets using an 80:20 split. The earlier 80% of the observations is reserved for model training, while the latest 20% is held out for evaluation.
The input features include calendar information, lagged electricity consumption, and a one-hour rolling mean. The prediction target is `Global_active_power`.
Chronological splitting is used to preserve the time order of the observations and evaluate model performance on a later period. The same test period will be used to compare the models.

