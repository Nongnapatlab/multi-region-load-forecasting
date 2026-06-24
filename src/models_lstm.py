"""Stateful LSTM wrapper that mirrors the sklearn fit/predict interface.

Encapsulates:
- MinMaxScaler fitting for both X and y
- Sequence construction (sliding window of length sequence_length)
- NaN-padding fix: the first (sequence_length - 1) rows that would otherwise
  be NaN are filled with the earliest available prediction instead of NaN.
  This prevents ensemble averaging from collapsing to NaN for those rows.
- None-model guard: if training data is too short to build even one sequence
  the model is a no-op and predict() returns np.nan everywhere.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from config import LSTM_PARAMS


class LSTMModel:
    """Sklearn-style wrapper for a single-layer LSTM regressor."""

    name = "LSTM"

    def __init__(self):
        self._model = None
        self._x_scaler = MinMaxScaler()
        self._y_scaler = MinMaxScaler()
        self._sequence_length: int = LSTM_PARAMS["sequence_length"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequences(self, X_scaled: np.ndarray, y_scaled: np.ndarray):
        seq_len = self._sequence_length
        Xs, ys = [], []
        for i in range(seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - seq_len:i])
            ys.append(y_scaled[i])
        return np.array(Xs), np.array(ys)

    def _build_predict_sequences(self, X_scaled: np.ndarray) -> np.ndarray:
        seq_len = self._sequence_length
        seqs = []
        for i in range(seq_len, len(X_scaled) + 1):
            seqs.append(X_scaled[i - seq_len:i])
        return np.array(seqs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LSTMModel":
        X_scaled = self._x_scaler.fit_transform(X_train)
        y_scaled = self._y_scaler.fit_transform(
            np.array(y_train).reshape(-1, 1)
        ).flatten()

        X_seq, y_seq = self._build_sequences(X_scaled, y_scaled)
        if len(X_seq) == 0:
            # Training set too short — model stays None
            return self

        model = Sequential([
            LSTM(
                LSTM_PARAMS["units"],
                input_shape=(X_seq.shape[1], X_seq.shape[2]),
            ),
            Dropout(LSTM_PARAMS["dropout"]),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        model.fit(
            X_seq,
            y_seq,
            epochs=LSTM_PARAMS["epochs"],
            batch_size=LSTM_PARAMS["batch_size"],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for every row in X.

        Fix: instead of NaN-padding the first (sequence_length - 1) rows,
        we forward-fill with the first real prediction so that ensemble
        averaging works across all rows without collapsing to NaN.
        """
        n = len(X)
        if self._model is None:
            return np.full(n, np.nan)

        X_scaled = self._x_scaler.transform(X)
        X_seq = self._build_predict_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.full(n, np.nan)

        preds_scaled = self._model.predict(X_seq, verbose=0).flatten()
        preds = self._y_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).flatten()

        # Align: preds[0] corresponds to row index (sequence_length - 1).
        # Forward-fill the warm-up rows with the first real prediction.
        result = np.empty(n)
        seq_len = self._sequence_length
        result[seq_len - 1:] = preds
        if seq_len > 1:
            result[: seq_len - 1] = preds[0]  # forward-fill warm-up rows
        return result