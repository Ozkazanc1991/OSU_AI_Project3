import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def get_LSTM_predictions(df, 
                         train_test_split = 0.8, 
                         window_size = 14, 
                         batch_size = 100, 
                         epochs = 5, 
                         layer1_nodes = 64, 
                         layer2_nodes = 64, 
                         layer3_nodes = 64,
                         verbose = 0) :

    # Calculate where and how much the data should be split for train and test.
    train_data_len   = int(len(df) * train_test_split)
    test_data_len    = len(df)-train_data_len

    # Features to include
    features = ["Open","High","Low","Volume"]

    # Scale the close prices to be in the range from 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled_df = scaler.fit_transform(df[features])
    y_scaled_df = scaler.fit_transform(df["Close"].to_frame())

    # Create the datasets
    X_train_data = X_scaled_df[:train_data_len]
    X_test_data  = X_scaled_df[train_data_len - window_size:]
    y_train_data = y_scaled_df[:train_data_len]
    y_test_data  = y_scaled_df[train_data_len - window_size:]

    X_train = []
    y_train = []

    # Populate the X_train and y_train, using sliding windows for X.
    for i in range(window_size, len(X_train_data)):
        X_train.append(X_train_data[i-window_size:i])
        y_train.append(y_train_data[i])
    
    # Convert the X_train and y_train to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # LSTMs expect the input data to be in a three-dimensional array with 
    # the following shape: [samples, window_size, features]
    # 
    # I am being very overtly clear here for my own understanding, this is new for me:
    number_of_sequences_in_data_set = X_train.shape[0]
    size_of_sliding_window = X_train.shape[1]
    number_of_features = len(features)

    X_train = np.reshape(X_train, (number_of_sequences_in_data_set,
                                   size_of_sliding_window,
                                   number_of_features))

    # Create the X_test and y_test datasets
    X_test = []
    y_test = y_test_data[int(train_data_len):int(train_data_len+test_data_len), :]

    for i in range(window_size, len(X_test_data)):
        X_test.append(X_test_data[i-window_size:i])

    # Convert the data to a numpy array
    X_test = np.array(X_test)

    # Reshape the data into the shape [samples, window_size, features]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], number_of_features))

    # Early stopping monitors the model’s performance on a validation set and stops 
    # training when the performance stops improving.
    # early_stopping = EarlyStopping(monitor='val_loss', 
    #                                patience=10, 
    #                                restore_best_weights=True)

    # Build the model
    model = Sequential()
    model.add(LSTM(units=layer1_nodes,  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=layer2_nodes,  return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=layer3_nodes, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    # Get the predictions
    predictions = scaler.inverse_transform(model.predict(X_test))

    return (predictions, model)

def plot_predictions(df, ticker_symbol, show_all) :
    
    # If we want to not show all, we need to drop rows where there are no predictions.
    if show_all : plot_df = df
    else        : plot_df = df.dropna(subset=['Predictions'])


    # Plotting using pandas' plot function
    plot_df['Close'].plot(figsize=(20, 10), title=f"{ticker_symbol} Predicted Closing Price vs. Actual Closing Price", label="Actual Closing Price")
    plot_df['Predictions'].plot(figsize=(10, 5), label="Predicted Closing Price")
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    return plt

def generate_buy_sell(df, predictions) : 

    # First populate the new features with blanks.
    df["Predictions"] = None
    df["Buy_Sell"]    = None

    # Create a None filled Numpy Array that is the size of the df minus
    # the size of the predictions Array
    none_array_size = int(len(df)-len(predictions))
    none_array = np.full((none_array_size,), None)

    # Then append predictions on the end, which should result in an array that's 
    # exactly the size we need to append to df.
    df["Predictions"] = np.append(none_array, predictions)

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

def score_the_model(df, start_date, end_date, verbose=False) :
    
    import pytz

    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    timezone = pytz.timezone('America/New_York')
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)

    # Filter the DataFrame based on the date range
    filtered_df = df[(df.index >= start_date) & (df.index < end_date)]

    initial_price = filtered_df.iloc[0]["Close"]
    final_actual_price = filtered_df.iloc[-1]["Close"]
    initial_buy_state = filtered_df.iloc[0]["Buy_Sell"]
    current_stock_count = 0
    current_balance = 0

    # Set the initial starting state
    if initial_buy_state == "BUY" :
        current_stock_count = 1
        current_balance = 0
    else : # initial_buy_state == "SELL"
        current_stock_count = 0
        current_balance = initial_price

    # Dropping the first day because we've already recorded our initial 
    # starting state
    filtered_df = filtered_df.drop(filtered_df.index[0])

    if verbose : 
        print("Initial Balance: ", initial_price)
        print("Initial Buy State: ", initial_buy_state)
        print()

    current_state = initial_buy_state
    current_balance = initial_price

    for date, row in filtered_df.iterrows():

        #print(date, row["Close"],row["Buy_Sell"])

        if row["Buy_Sell"] != current_state : 
            current_state = row["Buy_Sell"]
            if row["Buy_Sell"] == "BUY" :
                current_stock_count = current_balance / row["Open"]
                current_balance = 0
                if verbose: print(f"{date} Bought {current_stock_count} shares | Current Balance {current_balance}")
                
            else : # row["Buy_Sell"] == "SELL"
                current_balance = current_stock_count * row["Open"]
                if verbose: print(f"{date} Sold {current_stock_count} shares | Current Balance {current_balance}")
                current_stock_count = 0

    if current_balance == 0 : 
        # The money is still in the market and we need to cash out.
        current_balance = current_stock_count * filtered_df.iloc[-1]["Close"]

    if verbose:
        print()
        print("Final Balance: ", current_balance)
        print("Difference between actual close price and final balance: ", current_balance - final_actual_price)

    return (current_balance - final_actual_price)
