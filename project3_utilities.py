import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error



import warnings
warnings.filterwarnings('ignore')

def get_LSTM_predictions(df, 
                         train_test_split=0.8, 
                         window_size=14, 
                         batch_size=100, 
                         epochs=5, 
                         layer1_nodes=64, 
                         layer2_nodes=64, 
                         layer3_nodes=64,
                         verbose=0) :

    # Ensure data types
    train_test_split = float(train_test_split)
    window_size = int(window_size)
    batch_size = int(batch_size) 
    epochs = int(epochs)
    layer1_nodes = int(layer1_nodes)
    layer2_nodes = int(layer1_nodes)
    layer3_nodes = int(layer1_nodes)
    verbose = int(verbose)

    # Split the data into training and testing sets
    train_data_len = int(len(df) * train_test_split)
    test_data_len = len(df) - train_data_len

    # Features to include
    features = ["Open", "High", "Low", "Volume"]

    # Scale the features and target
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[features])
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    # Create the datasets
    X_train_data = scaled_features[:train_data_len]
    X_test_data = scaled_features[train_data_len - window_size:]
    y_train_data = scaled_target[:train_data_len]
    y_test_data = scaled_target[train_data_len - window_size:]

    X_train, y_train = [], []

    # Populate the X_train and y_train datasets using sliding windows
    for i in range(window_size, len(X_train_data)):
        X_train.append(X_train_data[i-window_size:i])
        y_train.append(y_train_data[i])

    # Convert the X_train and y_train to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape the data for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))

    # Create the X_test dataset
    X_test = []
    for i in range(window_size, len(X_test_data)):
        X_test.append(X_test_data[i-window_size:i])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

    y_test = y_test_data[window_size:]

    # Build the model
    model = Sequential()
    model.add(LSTM(units=layer1_nodes, return_sequences=True, input_shape=(window_size, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=layer2_nodes, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=layer3_nodes, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    # Get the predictions for the test data
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Predict the next day's close price
    last_sequence = scaled_features[-window_size:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    predicted_next_day_scaled = model.predict(last_sequence)
    predicted_next_day_close = target_scaler.inverse_transform(predicted_next_day_scaled)[0][0]

    return predictions, model, predicted_next_day_close, rmse

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

    initial_price = filtered_df.iloc[0]["Open"]
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

    total_profit = current_balance - final_actual_price
    final_actual_profit = final_actual_price - initial_price

    if verbose:
        print()
        print("Final Balance: ", current_balance)
        print("Difference between actual close price and final balance: ", current_balance - final_actual_price)

    return (total_profit, final_actual_price, final_actual_profit)
