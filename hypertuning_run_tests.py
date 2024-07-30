import pandas as pd
import yfinance as yf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import project3_utilities as p3utils

ticker_symbol = 'TSLA'
df = yf.Ticker(ticker_symbol).history(period='max')

test_splits = [0.75,0.8,0.85,0.9]
window_sizes = [7,14,30,100,365]
batch_sizes = [1,10,100,1000]
epoch_sizes = [1,5,10]
layer1_nodes = [16,32,64,128]
layer2_nodes = [16,32,64,128]
layer3_nodes = [16,32,64,128]

settings = []
for t in test_splits :
    for w in window_sizes : 
        for b in batch_sizes : 
            for e in epoch_sizes : 
                for l1 in layer1_nodes :
                    for l2 in layer2_nodes : 
                        for l3 in layer3_nodes :
                            settings.append( {"train_test_split" : t, 
                                            "window_size": w,    
                                            "batch_size": b, 
                                            "epochs": e, 
                                            "layer1_nodes": l1, 
                                            "layer2_nodes": l2, 
                                            "layer3_nodes": l3 } )

# Define the file name
filename = 'test_runs.csv'

# Open the file in write mode
with open(filename, mode='w', newline='') as file:
    # Create a DictWriter object
    writer = csv.DictWriter(file, fieldnames=settings[0].keys())
    
    # Write the header
    writer.writeheader()
    
    # Write the rows
    for row in settings:
        writer.writerow(row)

print(f"Data has been written to {filename}")

print(f"Starting {len(settings)} test runs.")

input("ready to go???")

scores = []
high_score = 0
high_score_settings = {}

for setting in settings : 
    unixtime = int(datetime.now().timestamp())

    (predictions, model) = p3utils.get_LSTM_predictions(df, **setting)

    #model.summary()
    #model.save(f'Saved_Models/{ticker_symbol}-Model-{unixtime}.keras')
    df = p3utils.generate_buy_sell(df, predictions)
    (df, profit_total) = p3utils.score_the_model(df)
    #df.to_csv(f'Saved_Predictions/{ticker_symbol}-Predictions-{unixtime}.csv')
    print(f"Run {unixtime} scored a {profit_total}")
    scores.append({"unixtime":unixtime, "score":profit_total} | setting)

    if profit_total > high_score : 
        high_score = profit_total
        high_score_settings = setting

for score in scores : print(score)
print(high_score_settings)
