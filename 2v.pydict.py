import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping  # Added for efficiency


# --- 1. Configuration and Constants ---
COMPANY = 'META'
TRAIN_START = dt.datetime(2014, 1, 1)
TRAIN_END = dt.datetime(2022, 1, 1)
TEST_START = dt.datetime(2022, 1, 2)  #  Test data should *not* overlap training data
TEST_END = dt.datetime.now()
PREDICTION_DAYS = 60
EPOCHS = 50  # Increased, but using EarlyStopping
BATCH_SIZE = 32
VERBOSE = 1  #  0 = silent, 1 = progress bar, 2 = one line per epoch


# --- 2. Data Loading Function ---
def load_data(company, start_date, end_date):
    """Loads stock data from yfinance."""
    return yf.download(company, start=start_date, end=end_date)


# --- 3. Data Preprocessing Function ---
def preprocess_data(data, prediction_days):
    """Scales data and prepares training sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train = []
    y_train = []

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler


# --- 4. Model Building Function ---
def build_lstm_model(input_shape):
    """Builds and compiles the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# --- 5. Prepare Test Data Function---
def prepare_test_data(train_data, test_data, scaler, prediction_days):
    """Preprocesses the test data for predictions, using the training data's scaler."""
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((train_data['Close'], test_data['Close']), axis=0)

    model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_input = model_input.reshape(-1, 1)
    model_input = scaler.transform(model_input)  # Use the *training* scaler

    x_test = []
    for i in range(prediction_days, len(model_input)):
        x_test.append(model_input[i - prediction_days:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test, actual_prices


# --- 6. Plotting Function ---
def plot_predictions(actual_prices, predicted_prices, company):
    """Plots the actual and predicted prices."""
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()


# --- 7. Main Execution Block ---

if __name__ == "__main__":  # Good practice for reusable code

    # Load training data
    train_data = load_data(COMPANY, TRAIN_START, TRAIN_END)

    # Preprocess training data
    x_train, y_train, scaler = preprocess_data(train_data, PREDICTION_DAYS)

    # Build the model
    model = build_lstm_model((x_train.shape[1], 1))

    # Train the model with EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[early_stopping])


    # Load test data
    test_data = load_data(COMPANY, TEST_START, TEST_END)

    # Prepare test data
    x_test, actual_prices = prepare_test_data(train_data, test_data, scaler, PREDICTION_DAYS)

    # Make predictions
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the results
    plot_predictions(actual_prices, predicted_prices, COMPANY)

    # --- Make a single future prediction ---
    # Prepare the last 'PREDICTION_DAYS' of *scaled* data
    real_data = scaler.transform(train_data['Close'].values.reshape(-1,1)) #Scale using the training data
    real_data = real_data[-PREDICTION_DAYS:]
    real_data = real_data.reshape(1, PREDICTION_DAYS, 1)

    # Predict and inverse transform
    future_prediction = model.predict(real_data)
    future_prediction = scaler.inverse_transform(future_prediction)
    print(f"Prediction for the next day: {future_prediction[0][0]}")
