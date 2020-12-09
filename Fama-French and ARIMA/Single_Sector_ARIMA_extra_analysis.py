#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:25:11 2020

@author: riku
"""

import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


#given ticker
#import stock data from YF

ticker = "WFC"
startDate = '2010-01-01'
endDate = '2019-01-01'

data = yf.download(ticker, startDate, endDate)

# experimenting with index column

#print(type(data)) #pandas dataframe
#print(data.columns.values.tolist())

#print(data.index)

#print(type(data.loc['2010-01-08']))

#print(data.loc['2010-01-08':'2010-02-01']["Close"])

#create time series plot
#data["Adj Close"].plot()
#plt.show()

a = data.loc['2010-01-08':'2014-06-08']["Close"]
b = data.loc['2010-01-08':'2014-09-08']["Close"]

a.dropna(inplace=True)
b.dropna(inplace=True)


#calculate log returns
returns = np.log(a) - np.log(a.shift(1))
log_returns = returns.dropna(axis=0)  # drop first missing row
print(log_returns.head())

#get logreturns

a.plot()
plt.show()

a = log_returns

#create autocorrelation plot of the time series
pd.plotting.autocorrelation_plot(a)
plt.show()

#create ARIMA model
# ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average

# A standard notation is used of ARIMA(p,d,q)

# p: The number of lag observations included in the model, also called the lag order.
# d: The number of times that the raw observations are differenced, also called the degree of differencing.
# q: The size of the moving average window, also called the order of moving average.

p = 5
d = 1
q = 0

# fit model
model = ARIMA(a, order=(p,d,q))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde') # ‘kde’ : Kernel Density Estimation plot
plt.show()
print(residuals.describe())

#plot predictive model

series = a

X = series.values
size = int(len(X) * 0.66) # why 2/3rds?
train, test, full = X[0:size], X[size:len(X)], X[0:len(X)]
history = [x for x in train]
print(len(history))
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

Y = b.values
extended = Y[size:len(Y)]

print(len(history))
future = []
future = model_fit.forecast(steps = 30)[0]

print("history type:")
print(type(history))
future_one_step = model_fit.forecast(steps = 1)[0]
print(future_one_step)
history.append(future_one_step)
print(len(history))

future2 = []
for t in range(30):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	future2.append(yhat)
	obs = future2[t]
	history.append(obs)

print(len(future))
print(len(future2))
#chopping block
# Threshold above which the line should be red
threshold = 60

#front = test[:threshold]
#back = predictions[threshold:]
#combined = np.append(front, back)
#
#aPandas = pd.Series(combined)
#bPandas = pd.Series(back)
#print(bPandas)
#ax = aPandas.plot()
#bPandas.plot(ax=ax)

#aPandas[aPandas > threshold].plot(color = 'red')

# plot
#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()

## plot
#front = test[:threshold]
#back = predictions[threshold:]
#print(future)
#print(future2)
#
##print(test)
##print("BREAK")
##print(front)
##print("BREAK")
##print(predictions)
##print("BREAK")
##print(back)
#
###with markers
##plt.plot(test, marker='o')
###plt.plot(front, marker='o')
##plt.plot(range(threshold,threshold+len(back)), back, color='red', marker='o')
##plt.plot(range(threshold+len(back),threshold+len(back)+len(future)), future, color='green', marker='o')
#
#threshold = threshold - 1
#
##without markers
#plt.plot(extended)
##plt.plot(front)
#plt.plot(range(threshold,threshold+len(back)), back, color='red')
#plt.plot(range(threshold+len(back),threshold+len(back)+len(future)), future, color='green')
#plt.plot(range(threshold+len(back),threshold+len(back)+len(future2)), future2, color='pink')
#
#plt.show()

##zoomed in
#plt.plot(extended)
##plt.plot(front)
#plt.plot(range(threshold,threshold+len(back)), back, color='red')
#plt.plot(range(threshold+len(back),threshold+len(back)+len(future)), future, color='green')
#plt.plot(range(threshold+len(back),threshold+len(back)+len(future2)), future2, color='pink')
#
#plt.xlim(100, 175)
#
#plt.show()
