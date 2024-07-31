import pandas as pd
import yfinance as yf
import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import project3_utilities as p3utils

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

#print(settings)
print("Number of test settings found: ", len(settings))

for setting in settings : 


    print("_____________________________________\n",setting)
    unixtime = int(datetime.now().timestamp())

    (predictions, model) = p3utils.get_LSTM_predictions(df, **setting)

    #model.summary()
    #model.save(f'Saved_Models/{ticker_symbol}-Model-{unixtime}.keras')
    df = p3utils.generate_buy_sell(df, predictions)
    testscore = p3utils.score_the_model(df, '2024-01-01','2024-06-30')
    #df.to_csv(f'Saved_Predictions/{ticker_symbol}-Predictions-{unixtime}.csv')
    print(f"Run {unixtime} scored a {testscore}")
    scores.append({"unixtime":unixtime, "score":testscore} | setting)

    if testscore > high_score : 
        high_score = testscore
        high_score_settings = setting

for score in scores : print(score)
print(high_score_settings)
