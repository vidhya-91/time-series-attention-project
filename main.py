import numpy as np
import pandas as pd

def generate_synthetic_time_series(
    n_points: int = 1500,
    noise_std: float = 1.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complex synthetic time series with trend,
    multiple seasonalities, structural break, and noise.

    Parameters
    ----------
    n_points : int
        Number of time steps.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Time series dataframe with time index and values.
    """
    np.random.seed(seed)
    t = np.arange(n_points)

    trend = 0.01 * t + 0.00001 * (t ** 2)
    seasonality_short = 5 * np.sin(2 * np.pi * t / 50)
    seasonality_long = 2 * np.sin(2 * np.pi * t / 200)
    structural_break = np.where(t > 900, 10, 0)
    noise = np.random.normal(0, noise_std, n_points)

    series = (
        trend
        + seasonality_short
        + seasonality_long
        + structural_break
        + noise
    )

    return pd.DataFrame({"time": t, "value": series()


# ===============================
# Attention LSTM Time Series Model
# ===============================

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# -------------------------------
# Create sequences
# -------------------------------
def create_sequences(series, window=30):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


# -------------------------------
# Build Attention LSTM model
# -------------------------------
def build_attention_lstm(window, units, learning_rate):
    inputs = Input(shape=(window, 1))

    lstm_out = LSTM(units, return_sequences=True)(inputs)

    attention_out = Attention()([lstm_out, lstm_out])

    context_vector = tf.reduce_mean(attention_out, axis=1)

    output = Dense(1)(context_vector)

    model = Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse"
    )
    return model


# -------------------------------
# Example Dataset (Synthetic)
# -------------------------------
np.random.seed(42)
data = np.sin(np.arange(0, 200)) + np.random.normal(0, 0.2, 200)
df = pd.DataFrame({"value": data})


# -------------------------------
# Scaling
# -------------------------------
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(df[["value"]]).flatten()


# -------------------------------
# Create sequences
# -------------------------------
WINDOW = 30
X_seq, y_seq = create_sequences(scaled_series, WINDOW)
X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)


# -------------------------------
# Hyperparameter tuning
# -------------------------------
best_rmse = float("inf")
best_params = None

for units in [32, 64]:
    for lr in [0.001, 0.0005]:

        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []

        for train_idx, test_idx in tscv.split(X_seq):
            model = build_attention_lstm(WINDOW, units, lr)

            model.fit(
                X_seq[train_idx],
                y_seq[train_idx],
                epochs=10,
                batch_size=32,
                verbose=0
            )

            preds = model.predict(X_seq[test_idx], verbose=0)
            rmse = mean_squared_error(
                y_seq[test_idx],
                preds,
                squared=False
            )
            rmses.append(rmse)

        avg_rmse = np.mean(rmses)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = {
                "units": units,
                "learning_rate": lr
            }


# -------------------------------
# Results
# -------------------------------
print("Best RMSE:", best_rmse)
print("Best Parameters:", best_params)




