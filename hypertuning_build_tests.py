import pandas as pd
import yfinance as yf
import numpy as np
import csv
import random
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

def get_test_case(settings) : 

    # Randomly select a setting from the settings
    random_index = random.randint(0, len(settings) - 1)

    # Pop the element at the randomly selected index
    selected_element = settings.pop(random_index)

    return (settings, selected_element)

def write_test_case(test_case, output_file_number, fieldnames) :
    
    with open(f"Test_Cases/testcases_{output_file_number}.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writerow(test_case)
    file.close()




current_file = 1
total_files = 7
fieldnames=settings[0].keys()


# Start the files, write the headers.
for i in range(1,total_files+1) :
    with open(f"Test_Cases/testcases_{i}.csv", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
    file.close()

# iterate through the settings, popping off a random setting and writing it 
# to a test file
while(settings) :

    # Randomly pull a test case from the list, reducing the size of list by one
    (settings, selected_setting) = get_test_case(settings)

    # Write the test case to the file
    write_test_case(selected_setting, current_file, fieldnames)

    if current_file == total_files : current_file = 1
    else : current_file += 1



