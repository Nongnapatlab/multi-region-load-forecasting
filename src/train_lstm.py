import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import LSTM_PARAMS


def create_sequences(X, y, sequence_length):
    Xs, ys = [], []
    for i in range(sequence_length, len(X)):
        Xs.append(X[i - sequence_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm_model(X_train, y_train):
    sequence_length = LSTM_PARAMS["sequence_length"]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_train)
    y_scaled = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    if len(X_seq) == 0:
        return None, x_scaler, y_scaler

    model = Sequential([
        LSTM(LSTM_PARAMS["units"], input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(LSTM_PARAMS["dropout"]),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(
        X_seq,
        y_seq,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model, x_scaler, y_scaler


def predict_lstm(model, x_scaler, y_scaler, X_all):
    if model is None:
        return np.full(shape=(len(X_all),), fill_value=np.nan)

    sequence_length = LSTM_PARAMS["sequence_length"]
    X_scaled = x_scaler.transform(X_all)

    X_seq = []
    for i in range(sequence_length, len(X_scaled) + 1):
        X_seq.append(X_scaled[i - sequence_length:i])
    X_seq = np.array(X_seq)

    if len(X_seq) == 0:
        return np.full(shape=(len(X_all),), fill_value=np.nan)

    preds_scaled = model.predict(X_seq, verbose=0).flatten()
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    padded = np.full(shape=(len(X_all),), fill_value=np.nan)
    padded[sequence_length - 1:] = preds
    return padded
