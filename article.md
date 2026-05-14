# Data Cleaning and Preprocessing

Data cleaning and preprocessing are vital for ensuring the accuracy and reliability of time series analysis. Time series data is often messy, with missing values, outliers, and irregular time intervals. Proper cleaning and preprocessing enhance data quality, reduce noise, and improve model performance. Without these steps, even the most advanced models can produce misleading results.

Missing values are common in time series data due to sensor failures, reporting delays, or data collection issues. These gaps can significantly affect model accuracy if not handled properly. Imputation methods, such as forward filling, backward filling, or interpolation, are commonly used to fill missing values. Forward filling uses the last observed value to fill gaps, while backward filling uses the next observed value. Interpolation estimates missing values based on surrounding observations, preserving the temporal pattern.

Outliers are another challenge, as they can distort trends, seasonality, and model forecasts. Outliers may occur due to data entry errors, system malfunctions, or genuine but rare events, such as stock market crashes or extreme weather conditions. Identifying outliers requires a deep understanding of the data context. Techniques like Z-score analysis, IQR (Interquartile Range), and visual inspection of time series plots help detect anomalies. Once identified, outliers can be treated by removing them, capping them within a predefined range, or replacing them with estimated values.

Time series data may also have irregular intervals due to missing timestamps or inconsistent sampling rates. Resampling techniques, such as aggregation and interpolation, help standardize the time intervals. Aggregation involves combining multiple observations into a single time period, such as converting hourly data into daily averages. Interpolation, on the other hand, estimates values for missing timestamps, creating a continuous time series.

Time zone adjustments are often necessary when dealing with global data sources. Inconsistent time zones can lead to misalignment of observations, affecting the accuracy of trend and seasonality analysis. Converting all timestamps to a common time zone standardizes the data, ensuring temporal consistency. This is particularly important for financial markets, where trading hours differ across regions.

After cleaning the data, transformations are applied to enhance model performance and interpretability. Log transformations stabilize variance, while differencing removes trends and seasonality, making the series stationary. Normalization and standardization ensure that all variables have the same scale, preventing models from being biased towards features with larger magnitudes.

Data cleaning and preprocessing is step 1 for accurate time series analysis. By effectively handling missing values, outliers, and irregular intervals, you ensure that the data accurately reflects the underlying patterns. The next section explores feature engineering, which builds on these clean datasets to create informative variables that enhance model performance.

# Time Series Manipulation with Pandas in Python

Pandas has built-in tools for analyzing time series data, such as shifting, windowing, resampling, and imputing missing values. While data engineering is not the most exciting part of time series analysis, it is crucial for reliable and accurate forecasting.

## Shifting Data

Shifting involves moving data forward or backward in time. It is commonly used to create lagged features or compare values across different time periods.

## Creating Lagged Features

Lagged features are essential for capturing temporal dependencies in time series data. They allow you to compare the value today against the value from a previous time step, which is particularly useful for creating lagged variables in machine learning models.

    # Example Time Series Data
np.random.seed(42) data = pd.Series(np.random.randn(10), index=pd.date_range("2023-01-01", periods=10))

    # Creating Lagged Features
data_lag1 = data.shift(1) # Lag by 1 period data_lag2 = data.shift(2) # Lag by 2 periods

print("Original Data:\n", data) print("Lagged by 1 Period:\n", data_lag1) print("Lagged by 2 Periods:\n", data_lag2)

## Windowing (Rolling Statistics)

Windowing calculates statistics over a rolling window, such as moving averages or standard deviations. It smooths data and highlights trends.

Rolling Mean:
    # Rolling Mean with Window Size of 3
rolling_mean = data.rolling(window=3).mean() print("Rolling Mean (Window=3):\n", rolling_mean)

The rolling mean is the most common windowing technique, but rolling windows can also be used with other aggregations.

Rolling Standard Deviation:
    # Rolling Standard Deviation (Volatility)
rolling_std = data.rolling(window=3).std() print("Rolling Standard Deviation (Window=3):\n", rolling_std)

Calculating rolling standard deviation is useful for analyzing volatility, especially in financial time series.

## Resampling

Resampling changes the frequency of time series data, either by aggregating data (downsampling) or by interpolating data (upsampling).

Downsampling:
    # Example Time Series Data (Hourly)
date_rng = pd.date_range(start='2023-01-01', end='2023-01-02', freq='H') data_hourly = pd.Series(np.random.randn(len(date_rng)), index=date_rng)

    # Downsample to Daily Mean
data_daily = data_hourly.resample('D').mean() print("Daily Mean (Downsampled):\n", data_daily)

Upsampling:
    # Upsample to 30-Minute Frequency and Interpolate
data_half_hourly = data_hourly.resample('30T').interpolate(method='linear') print("Upsampled and Interpolated (30 Min Frequency):\n", data_half_hourly)

## Imputing Missing Values

Machine learning models are sensitive to missing values. Pandas provides several strategies for imputing missing values, such as forward fill, backward fill, and mean imputation.

Forward Fill (or Back Fill):
    # Introducing NaN values
data_with_nans = data.copy() data_with_nans.iloc[3] = np.nan data_with_nans.iloc[7] = np.nan

    # Forward Fill
data_ffill = data_with_nans.ffill() print("Forward Filled Data:\n", data_ffill)

    # Backward Fill
data_bfill = data_with_nans.bfill() print("Backward Filled Data:\n", data_bfill)

Forward fill is particularly useful when working with sensor data where the most recent observation is carried forward until a new reading is available.

Mean Imputation:
    # Mean Imputation
data_mean_impute = data_with_nans.fillna(data_with_nans.mean()) print("Mean Imputation:\n", data_mean_impute)

Mean imputation is useful when gaps are short or when the data is relatively stable.

## Combining Techniques

In practice, time series analytics often require a combination of shifting, windowing, resampling, and imputing to prepare the data for modeling. This flexibility makes Pandas a powerful tool for time series preprocessing.

## Takeaway

Most time series libraries have features for shifting, windowing, resampling, and imputing. The advantage of using Pandas is its seamless integration with dataframes, allowing you to perform these transformations without needing a specialized library. This makes Pandas an excellent choice for data preprocessing in time series analysis.

# Feature Engineering for Time Series Forecasting in Python

Feature engineering is the process of creating additional input features from raw time series data to improve the performance of predictive models. Unlike static datasets, time series data has unique temporal properties---patterns like trends, seasonality, and lag relationships---that can be extracted and transformed into valuable features. This chapter explores feature engineering techniques for time series, including scaling, differencing, derivatives, and memory embedding.

Time series forecasting is not just about feeding raw data into a model. Transforming the data to highlight important patterns or trends can:

- Improve Model Accuracy by adding relevant features.

- Reveal Hidden Patterns like rates of change or lagged effects.

- Help Models Learn Temporal Dependencies, especially when using non-sequential models like regression.

Good feature engineering ensures models are equipped to learn both short-term and long-term relationships in the data.

## Scaling Values

Time series values often vary in magnitude, and unscaled data can hinder the performance of models, especially those that rely on gradient descent (e.g., neural networks). Scaling methods include:

- Min-Max Scaling: Rescales values to a specific range, e.g., \[0, 1\].

- Standardization (Z-score): Rescales values to have a mean of 0 and standard deviation of 1.

Python Example for Scaling:
from sklearn.preprocessing import MinMaxScaler, StandardScaler import numpy as np

    # Example Time Series
y = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)

    # Min-Max Scaling
scaler = MinMaxScaler() y_minmax = scaler.fit_transform(y)

    # Standardization
scaler_std = StandardScaler() y_std = scaler_std.fit_transform(y)

print("Min-Max Scaled:", y_minmax.flatten()) print("Standardized:", y_std.flatten())

When to Use:
- Use Min-Max Scaling for data where the range matters.

- Use Standardization when the scale is unknown or when working with models sensitive to variance.

## Looking at Changes in Values

Instead of analyzing absolute values, focusing on changes can remove trends and reveal stationarity. Differencing calculates the difference between consecutive values:

Python Example for Differencing:
    # Example Time Series
y = pd.Series([10, 12, 15, 19, 24])

    # First Difference
y_diff = y.diff().dropna() print("First Difference:", y_diff.values)

First-order differencing removes trends to make the data stationary and highlights short-term changes in the series. However, differencing reduces the length of the dataset, and excessive differencing may lead to a loss of long-term information. Analysts must balance improved stationarity against potential data loss.

## Derivatives: Rate of Change and Acceleration

Derivatives measure the rate of change in a time series, which can highlight momentum or acceleration patterns.

- First Derivative: Measures the rate of change.

- Second Derivative: Measures the rate of change of the rate of change (acceleration).

Python Example for Derivatives:
import numpy as np

    # Example Time Series
y = np.array([10, 12, 15, 19, 24])

    # First Derivative (Rate of Change)
dy = np.gradient(y) print("First Derivative (Rate of Change):", dy)

    # Second Derivative (Acceleration)
d2y = np.gradient(dy) print("Second Derivative (Acceleration):", d2y)

## Embedding Prior Values: Building \"Memory\"

Embedding previous observations as features allows models to \"remember\" past values. This is especially important for models that do not inherently capture temporal dependencies (e.g., regression).

Python Example for Lagged Features:
    # Simulated Time Series Data
np.random.seed(42) data = pd.Series(np.cumsum(np.random.randn(200))) # Random walk time series

    # Create Features: Lagged Values and Rate of Change
df = pd.DataFrame({ 'value': data, 'lag_1': data.shift(1), 'lag_2': data.shift(2), 'rate_of_change': data.diff() }).dropna()

Caution: Adding too many lagged features can cause overfitting.

## Other Common Feature Engineering Techniques

- Rolling Statistics: Calculate rolling means, variances, or medians over a window to smooth the series.

rolling_mean = y.rolling(window=3).mean()

- Extracting Seasonality: Decompose a series into trend, seasonal, and residual components.

from statsmodels.tsa.seasonal import seasonal_decompose result = seasonal_decompose(y, period=12) result.seasonal.head()

- Fourier Transforms: Identify dominant frequencies in seasonal data.

- Time-Based Features: Extract calendar-related features like month, day of the week, or hour to capture seasonality.

df.index = pd.date_range(start="2023–01", periods=len(df), freq="M") df_features = pd.DataFrame({"month": df.index.month, "year": df.index.year}) print(df_features.head())

## Next Steps

Feature engineering transforms raw time series data into meaningful representations, enabling models to learn richer temporal patterns. Techniques like scaling, differencing, derivatives, and memory embedding improve predictive accuracy.

Example - Beehive Analysis
- Lagged features: Yesterday's weight affects today's weight.

- Rate of Change: First derivative of weight to detect honey production speed.

Python Example:
df['weight_lag1'] = df['weight'].shift(1) df['weight_derivative'] = df['weight'].diff() df.head()

Pandas makes these transformations straightforward for time series datasets, enabling effective feature engineering and enhancing model accuracy.

# Exponential Smoothing vs. Moving Average for Time Series Analysis

In time series forecasting, smoothing methods help filter noise and reveal underlying patterns. Two of the most widely used techniques are moving averages and exponential smoothing. Both methods assign weights to past observations, but they differ fundamentally in how they handle recent vs. older data. Let's explore the differences between these approaches and their weighting schemes.

# Moving Average: Equal Weight for Recent Observations

![image](img/smoothing.png)

A moving average (MA) smooths a time series by averaging a fixed number $N$ of past observations. In a simple moving average the last $N$ data points are given equal weight. The weight assigned to each of the $N$ observations is $\frac{1}{N}$. We discard (or ignore) older data points outside our chosen window.

For example, in a 5-period moving average, each of the last five observations receives a weight of 0.2, while older observations have a weight of 0. This can be seen in the black dashed line in the graph.

Moving Average is easy to compute and interpret. It works well for stationary time series with minimal trend. It treats all included observations equally and ignores recency effects. This can make the trend slower to respond to sharp changes. Any trend that is longer than our window will not be picked up/used by the moving average.

# Exponential Smoothing: Prioritizing Recent Data

Unlike moving averages, exponential smoothing (ES) assigns exponentially decreasing weights to older observations. The weight for each observation is determined by a smoothing factor $\alpha$ where $0 < \alpha < 1$.

- $\alpha$ determines how quickly older observations lose influence.

- A higher $\alpha$ places more emphasis on recent observations, while a lower $\alpha$ spreads the influence over a longer range of past data.

Impact of Different $\alpha$ Values
- $\alpha = 0.3$ (Red Line): The weight for recent data decays quickly, giving the most importance to the latest observation.

- $\alpha = 0.2$ (Orange Line): Weights decay more gradually, spreading influence over a broader history.

- $\alpha = 0.1$ (Yellow Line): Weights decline very slowly, incorporating long-term patterns.

As shown in the graph, lower values of $\alpha$ resemble a long-tailed moving average, while higher values behave more like a weighted moving average with stronger emphasis on the most recent data.

Exponential Smoothing is more flexible than moving averages. The weights decline smoothly, rather than being cut off at a fixed (and usually arbitrary) window. We can extend exponential smoothing to double and triple exponential smoothing to handle trends and seasonality.

On the flip side, exponential Smoothing requires us to choose an appropriate $\alpha$. There is no single statistical test that can tell us the best value for $\alpha$ so we may have to try several values to see which one performs the best. Out of the box, it doesn't handle seasonality; we have to use extensions to account for that.

# Choosing Between Moving Average and Exponential Smoothing

In general, moving averages work well when trends are minimal, and you want a simple smoothing technique. Exponential smoothing is a better choice when recent data is more informative and trends need to be accounted for.

# The Effect of N on Lag in Moving Averages

One of the most critical drawbacks of moving averages is lag---the delay introduced in detecting changes in the underlying time series. The larger the window size $N$, the greater the lag.

Understanding Lag in Moving Averages Since a moving average smooths a time series by averaging over the last $N$ observations, it effectively delays the recognition of changes in the data. When a trend shifts upward or downward, the moving average will only gradually adjust to the new trend because it incorporates past values that no longer represent the current direction of the series.

Small N (e.g., N=3)
- The moving average reacts more quickly to changes.

- The curve remains close to the actual data points.

- There is less lag, but also more sensitivity to noise.

Large N (e.g., N=10)
- The moving average smooths the series more aggressively.

- It takes longer to detect an upward or downward shift.

- The lag is more pronounced, delaying trend recognition.

Why Lag Matters For decision-making, lag can be problematic, especially in applications where quick responses are crucial:

- In financial trading, a large $N$ might cause traders to miss early signs of a price reversal.

- In demand forecasting, too much lag delays inventory adjustments, leading to stockouts or overstock.

- In industrial monitoring, a slow-moving average might fail to quickly detect a sudden drop in machine efficiency.

# Visualizing Lag: The Impact of N

If we overlay multiple moving averages with different values of $N$, we can see that:

- Smaller $N$ follows the data closely but remains noisy.

- Larger $N$ smooths fluctuations but lags behind changes.

This demonstrates a fundamental trade-off:

- Lower lag (small $N$) = more responsive but noisier predictions.

- Higher lag (large $N$) = smoother but slower adaptation to new trends.

Comparing with Exponential Smoothing Exponential smoothing also introduces lag, but because recent observations receive more weight, it reacts faster than a moving average of equivalent smoothing power. By adjusting $\alpha$, analysts can control the trade-off between responsiveness and lag, much like changing $N$ in a moving average.

![image](img/smoothing_chart.png)

Choosing N Wisely When selecting $N$ for a moving average, the decision depends on the desired balance between smoothness and responsiveness:

- For short-term trend detection → Use a smaller $N$ (e.g., 3--5).

- For long-term stability and noise reduction → Use a larger $N$ (e.g., 10--20).

- If lag is unacceptable → Consider switching to exponential smoothing with a moderate $\alpha$, which provides less lag than a moving average for the same level of noise reduction.

Both moving averages and exponential smoothing are useful techniques for time series forecasting, but they serve different purposes. The key takeaway is that exponential smoothing adapts to recent changes better while moving averages offer a stable, straightforward approach to smoothing. The choice of $\alpha$ in exponential smoothing determines the balance between responsiveness and stability, allowing users to fine-tune their model based on the characteristics of their data.

## Key Takeaways

- Improve Model Accuracy by adding relevant features.
- Reveal Hidden Patterns like rates of change or lagged effects.
- Help Models Learn Temporal Dependencies, especially when using non-sequential models like regression.
- Min-Max Scaling: Rescales values to a specific range, e.g., \[0, 1\].
