#Stock Analysis - OSU Artificial Intelligence - Project #3

***SUMMARY:***
  ***In this project we utilized Kaggle TSLA Stock Price Data. We will be using LSTM nodes which have the ability to hold information over long-periods of time. In our case, the information we would be storing would be the TSLA closing stock price. Our analysis will also utilize time-series forecasting.***

***OBJECTIVES:***
  ***Our main objectives for this project includes predicting the TSLA stock price through LSTM Neural Network Nodes. In performing our test and analysis, we will be utilizing these nodes and their storage capabilities to accurately predict our stock prices.***

***OUR DATASET:*** 
  ***All Financial Data comes from Yahoo Finance.***
  ***In creating our LSTM Model, we also created an API Key to allow for pulling of headline information found in Yahoo Finance.***

***STATISTICS / GRAPHICS:***
  ***In creating our LSTM Model, we utilized ‘Adam’ as our optimizer. We then created our train_test_split and plotted our predictions. In plotting our predictions, we followed up by creating Buy / Sell predictions.***
  ***Our code enables comparisons which compare buying / holding / selling for each iteration. (With each decision, what would happen if we chose a different option.)***
  ***As our model cycles through buying and selling iterations, the profit or loss is summed in a cumulative balance.***

***MISC NOTES:***
  ***In creating our LSTM Model, we enabled a user function that allows for input of a 'stock-ticker' letting the user choose what stock they would like to see information for. This increases the functionality and capabilities ten-fold.***
  ***Our model hypertunes / trains our information by years.***
  ***Our dataset is trained on three node layers with our activation being 'relu'.***

Programming Language Used: Python.

Packages Used: Anaconda, Pandas, Numpy, Matplotlib.pyplot, sklearn.preprocessing - MinMaxScaler,  tensorflow.keras.models - Sequential, and tensorflow.keras.layers - LSTM, Dense.

Tools Used: Jupyter Notebook (IPython).

Credit & Contributors: ballen614, Ozkazanc1991, 2Hail.
