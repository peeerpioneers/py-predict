# py predict
#------------
pip install numpy
pip install Matplotlib
pip install --upgrade tensorflow
pip install -U scikit-learn
pip install pandas-datareader
#pip install statsmodels
#pip install PyPortfolioOpt

#-------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import PredefinedSplit
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
#--------------------------------------------------
import yfinance as yf  # Install with: pip install yfinance
import datetime as dt

company = 'META'  # Use the correct ticker symbol for Meta
start = dt.datetime(2014, 1, 1)
end = dt.datetime(2022, 1, 1)

data = yf.download(company, start=start, end=end)

print(data.head())
print(data.tail())
#--------------------------------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# how many days we want to look at the past to predict
prediction_days = 60

# defining two empty lists for preparing the training data
x_train = []
y_train = []

# we are counting from the 60th index to the last index
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#-----------------------------------------------------
model.compile(optimizer='adam', loss='mean_squared_error')
# fit the model in the training data
model.fit(x_train, y_train, epochs=25, batch_size=32)
#-----------------------------------------------------
model = Sequential()
# specify the layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# this is going to be a prediction of the next closing value
model.add(Dense(units=1))
#----------------------------------------------------
model.compile(optimizer='adam', loss='mean_squared_error')
# fit the model in the training data
model.fit(x_train, y_train, epochs=25, batch_size=32)
#---------------------------------------------------
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'],test_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset)- len(test_data) - prediction_days:].values
# reshaping the model
model_input = model_input.reshape(-1, 1)
# scaling down the model
model_input = scaler.transform(model_input)
#----------------------------------------------------
x_test = []
for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# plot the test Predictions
plt.plot(actual_prices, color="black", label=f"Actual{company} price")
plt.plot(predicted_price, color='green', label="Predicted {company} Price")
plt.title(f"{company} Share price")
plt.xlabel('Time')
plt.ylabel(f'{company} share price')
plt.legend
plt.show()
#----------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf # Or your chosen data source
import datetime as dt

# --- Data Loading (from your previous working examples) ---
company = 'META'
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
test_data = yf.download(company, start=test_start, end=test_end)

# --- Data Preprocessing ---
# 1. Select the 'Close' column (or whichever feature you're predicting).
data = test_data['Close'].values  # Get the closing prices as a NumPy array

# 2. Reshape for scaling.  MinMaxScaler expects 2D input (samples, features).
data = data.reshape(-1, 1)

# 3. Create and fit the scaler.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- Preparing real_data ---
# 4.  Get the last 60 days of *scaled* data.  This is *crucial*.
#     You need to predict on data that is in the same format (scaled)
#     as the data the model was trained on.
sequence_length = 60  # Or whatever your model was trained on
real_data = scaled_data[-sequence_length:]

# 5.  Reshape for the LSTM (as in the original example).
real_data = real_data.reshape(1, sequence_length, 1)  # (1 sample, 60 timesteps, 1 feature)

# --- Prediction ---
#  (Assuming your 'model' is already trained)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
#---------------------------------------------------------

