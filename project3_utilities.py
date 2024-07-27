import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_LSTM_predictions(df, ticker_symbol, train_test_split, model, window_size) :
    
    # Features to consider
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Calculate where and how much the data should be split for train and test.
    train_data_len   = int(len(df) * train_test_split)
    test_data_len    = len(df)-train_data_len
    train_data_start = 0
    test_data_start  = int(train_data_len)

    # Extract the close prices
    close_prices = df['Close'].values
    close_prices = close_prices.reshape(-1, 1)

    # Remove the close price and adj close price from the original data.
    #df = df.drop(['Close', 'Adj Close'], axis=1)
    #display(df.head())

    # Scale the close prices to be in the range from 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)

    # Create the training dataset
    train_data = scaled_close_prices[:train_data_len]

    X_train = []
    y_train = []

    # Populate the X_train and y_train, using sliding windows for X.
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the X_train and y_train to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # LSTMs expect the input data to be in a three-dimensional array with 
    # the following shape: [samples, window_size, features]
    # 
    # I am being very overtly clear here for my own understanding, this is new for me:
    number_of_sequences_in_data_set = X_train.shape[0]
    size_of_sliding_window = X_train.shape[1]
    number_of_dataframe_features = 1 # in this case, because we only have closing price.

    X_train = np.reshape(X_train, (number_of_sequences_in_data_set,
                                size_of_sliding_window,
                                number_of_dataframe_features))

    # Create the testing dataset
    test_data = scaled_close_prices[train_data_len - window_size:]

    # Create the x_test and y_test datasets
    X_test = []
    y_test = close_prices[int(train_data_len):int(train_data_len+test_data_len), :]

    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])

    # Convert the data to a numpy array
    X_test = np.array(X_test)

    # Reshape the data into the shape [samples, window_size, features]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)

    # Get the predictions
    predictions = scaler.inverse_transform(model.predict(X_test))

    return (predictions, model)

def plot_predictions(df, ticker_symbol, train_test_split, predictions, show_all) :
    
    # Prepare the data for the X-Axis
    train_data_len = int(len(df) * train_test_split)
    train_dates = df['Date'][:train_data_len]
    test_dates = df['Date'][train_data_len:]

    # Plot the data
    train = df[:train_data_len]
    actual = df[train_data_len:]
    actual['Predictions'] = predictions

    plt.figure(figsize=(16,8))
    plt.title(ticker_symbol)
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')

    if show_all : plt.plot(train_dates, train['Close'])
    
    plt.plot(test_dates, actual[['Close', 'Predictions']])

    if show_all : plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
    else : plt.legend(['Actual', 'Predictions'], loc='lower right')
    
    return plt

def generate_buy_sell(df, predictions) : 

    # First populate the new features with blanks.
    df["Predictions"] = None
    df["Buy_Sell"]    = None

    # Then, fill the new predictions feature with the prediction values, but
    # "fill from the bottom"
    insertion_point = len(df)-len(predictions)
    df.loc[insertion_point:insertion_point+len(predictions)-1, "Predictions"] = predictions

    # Next calculate and fill the buy/sell state into the buy_sell feature.
    
    # Calculate the difference between the current row and the previous row
    df['Close_Difference'] = df['Close'].diff()

    def buy_or_sell(diff):
        if pd.isnull(diff):  # Handle the first row or any NaN values
            return None
        elif diff > 0:
            return "BUY"
        else:
            return "SELL"

    # Apply the function to the 'Difference' column to create the 'Action' column
    df["Buy_Sell"] = df['Close_Difference'].apply(buy_or_sell)
    df = df.drop(columns=['Close_Difference'])

    return df

def score_the_model(df) :

    def calculate_profit(row):
        if row['Buy_Sell'] == 'BUY' : return row['Close'] - row['Open']
        else : return 0

    # Apply the function to each row
    df['Profit'] = df.apply(calculate_profit, axis=1)

    return (df, df['Profit'].sum())
