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

# Open the file in write mode

for i in range(1,4) : 
    with open(f"testcases_{i}.csv", mode='w', newline='') as file:
        # Create a DictWriter object
        writer = csv.DictWriter(file, fieldnames=settings[0].keys())
        
        # Write the header
        writer.writeheader()
        
        # Write the rows
        for row in settings:
            writer.writerow(row)

