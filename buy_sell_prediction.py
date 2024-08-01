import pandas as pd
import yfinance as yf
import numpy as np
import csv
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import project3_utilities as p3utils

import warnings
warnings.filterwarnings('ignore')

ticker_symbol = input('Enter Ticker Symbol: ').upper()
df = yf.Ticker(ticker_symbol).history(period='max')

(predictions, model, predicted_next_day_close, rmse) \
    = p3utils.get_LSTM_predictions(df=df, 
                                   train_test_split=0.85, 
                                    window_size=365,
                                    batch_size=10,
                                    epochs=1,
                                    verbose=1,
                                    layer1_nodes=16,
                                    layer2_nodes=16,
                                    layer3_nodes=128)

df = p3utils.generate_buy_sell(df, predictions)

print(f"\nTomorrow you should {df.iloc[-1]['Buy_Sell']} {ticker_symbol} stock.\n")