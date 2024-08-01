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

ticker_symbol = 'TSLA'
df = yf.Ticker(ticker_symbol).history(period='max')

scores = []
high_score = 0
high_score_settings = {}

# Load the test file passed in from the command line
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help="The name of the file to process")
args = parser.parse_args()
filename = args.filename

print("Loading test file: ", filename)

# Load the file into a list
with open(filename, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    settings = [row for row in reader]
    file.close()

#print(settings)
print("Number of test settings found: ", len(settings))

# Check if the scores file exists, and if not, write out the file with column headers.
scores_file_exists = False
if os.path.exists("Test_Cases/all_scores.csv") : scores_file_exists = True


for setting in settings : 
    print(setting)

    unixtime = int(datetime.now().timestamp())

    (predictions, model, tomorrows_predicted_close, rmse) = p3utils.get_LSTM_predictions(df, **setting, verbose=1)

    #model.summary()
    df = p3utils.generate_buy_sell(df, predictions)
    (testscore, final_actual_price, final_actual_profit)  = p3utils.score_the_model(df, '2024-01-01','2024-07-31')
    print(f"Run {unixtime} scored a {testscore} with an RMSE of {rmse}")

    setting['profit'] = testscore
    setting['rmse'] = rmse

    with open(f"Test_Cases/all_scores.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, settings[0].keys())
        if not scores_file_exists : writer.writeheader()
        writer.writerow(setting)
    file.close()


for score in scores : print(score)
print(high_score_settings)
