"""Stateful LSTM wrapper that mirrors the sklearn fit/predict interface.

Encapsulates:
- MinMaxScaler fitting for both X and y
- Sequence construction (sliding window of length sequence_length)
- Deterministic seeding (numpy/random/tensorflow) so repeated runs on the
  same data produce the same predictions.
- Warm-up rows (the first sequence_length - 1 rows of any predict() call,
  which don't have enough history to form a sequence) are left as NaN.
  ensemble.py's weighted_average_ensemble renormalizes weights per-row, so
  these rows simply fall back to the other models instead of being
  dragged toward a copy-pasted constant.
- None-model guard: if training data is too short to build even one sequence
  the model is a no-op and predict() returns np.nan everywhere.
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from config import LSTM_PARAMS, RANDOM_STATE


def _set_deterministic_seed(seed: int = RANDOM_STATE) -> None:
    """Fix all relevant RNGs so LSTM training is reproducible across runs.
    Without this, weights initialize differently every run and predictions
    (and therefore MAPE) will vary run-to-run even with identical data."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
        _set_deterministic_seed()

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

        The first (sequence_length - 1) rows have no real LSTM prediction
        (there isn't enough history yet to form a sequence) — they are left
        as NaN rather than forward-filled with the first real prediction.

        Forward-filling looks convenient (no NaNs to handle downstream) but
        is misleading: with a short test set (e.g. 288 rows, sequence_length
        24), forward-fill can mean *every single realized row* in a day
        gets the same constant value repeated, which is not really "the
        LSTM's prediction" for those timestamps — it's one prediction
        copy-pasted. That constant then drags the ensemble toward a flat
        line regardless of how much actual demand actually varied that day.

        Leaving these as NaN and letting weighted_average_ensemble
        renormalize weights per-row (see ensemble.py) means rows without
        enough LSTM history simply fall back to XGBoost+LightGBM only,
        which is the honest answer when LSTM genuinely has nothing to
        contribute yet.
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
        # Rows before that have no real prediction — leave them as NaN.
        result = np.full(n, np.nan)
        seq_len = self._sequence_length
        result[seq_len - 1:] = preds
        return result