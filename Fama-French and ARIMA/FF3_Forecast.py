#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:03:58 2020

@author: riku
"""

# Pandas to read csv file and other things
import pandas as pd
import yfinance as yf 
from itertools import product 
import statsmodels.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta 

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from tqdm import tqdm_notebook

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)

#Predict Fama French 3 Factors using ARIMA Model.

#1. Get FF3 Data
def get_fama_french():
    # Now open the CSV file
    ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows = 3, index_col = 0)
    # We want to find out the row with NULL value
    # We will skip these rows
    ff_row = ff_factors.isnull().any(1).nonzero()[0][0]
    # Read the csv file again with skipped rows
    ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows = 3, nrows = ff_row, index_col = 0)
    # Format the date index
    ff_factors.index = pd.to_datetime(ff_factors.index, format= '%Y%m')
    # Format dates to end of month
    ff_factors.index = ff_factors.index + pd.offsets.MonthEnd()
    # Convert from percent to decimal
    ff_factors = ff_factors.apply(lambda x: x/ 100)
    return ff_factors

ff_data = get_fama_french()
print(ff_data.head())

#Isolate each factor
Rm_Rf = ff_data['Mkt-RF']
SMB = ff_data['SMB']
HML = ff_data['HML']

#data = Rm_Rf
#data = SMB
data = HML

##time plot
#data.plot()
#plt.show()
#
#create autocorrelation plot of the time series
plot_pacf(data)
plot_acf(data)
plt.show()

# Augmented Dickey-Fuller test, d = 0
ad_fuller_result = adfuller(data)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try: 
            model = SARIMAX(exog, order=order).fit(disp=-1)
        except:
            continue
            
        #aic = model.aic
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


##copied AIC
#ps = range(0, 4, 1)
#d = 0
#qs = range(0, 4, 1)
#
## Create a list with all possible combination of parameters
#parameters = product(ps, qs)
#parameters_list = list(parameters)
#
#order_list = []
#
#for each in parameters_list:
#    each = list(each)
#    each.insert(1, 0)
#    each = tuple(each)
#    order_list.append(each)
#    
#result_df = optimize_ARIMA(order_list, exog=data)
#
#print(result_df)
#
#p = 2
#d = 0
#q = 1
#
## fit model
#model = ARIMA(data, order=(p,d,q))
#model_fit = model.fit(disp=0)
#output = model_fit.forecast()
#print(output[0])
    

p = 2
d = 0
q = 2

# fit model
model = ARIMA(Rm_Rf, order=(p,d,q))
model_fit = model.fit(disp=0)
output = model_fit.forecast()
Rm_Rf_predict= output[0]
print(output[0])
print(model_fit.params)

p = 2
d = 0
q = 1

# fit model
model = ARIMA(SMB, order=(p,d,q))
model_fit = model.fit(disp=0)
output = model_fit.forecast()
SMB_predict= output[0]
print(output[0])
print(model_fit.params)

p = 0
d = 0
q = 1

# fit model
model = ARIMA(HML, order=(p,d,q))
model_fit = model.fit(disp=0)
output = model_fit.forecast()
HML_predict= output[0]
print(output[0])
print(model_fit.params)

print(len(ff_data))

#ff_predict = pd.DataFrame({"Mkt-RF":Rm_Rf_predict, 
#                           "SMB":SMB_predict,  
#                           "HML":HML_predict,
#                           "RF":[0]}) 
    
ff_predict = {"Mkt-RF":Rm_Rf_predict, 
              "SMB":SMB_predict,  
              "HML":HML_predict,
              "RF":[0]}


#print(ff_predict)

last_date = ff_data.iloc[[-1]].index
last_date = last_date + timedelta(days = 31)

ff_forcast = ff_data.append(pd.DataFrame(ff_predict, index=last_date))
#ff_forcast = ff_data.append(ff_predict)
print(len(ff_forcast))
print(ff_forcast.tail())


def get_price_data(ticker, start, end):
    temp = yf.download(ticker, start, end)
    temp.dropna(inplace=True)
    price = temp[temp['Adj Close'] > 1]  
    price = price['Adj Close'] # keep only the Adj Price col
    return price

def get_return_data(price_data, period = "M"):
    # Resample the data to monthly price
    price = price_data.resample(period).last()
    # Calculate the percent change
    ret_data = price.pct_change()[1:]
    # convert from series to DataFrame
    ret_data = pd.DataFrame(ret_data)
    # Rename the Column
    ret_data.columns = ['portfolio']
    return ret_data

def run_reg_model(ticker,start,end):
    # Get FF data
    ff_data = ff_forcast
    ff_last = ff_data.index[ff_data.shape[0] - 1].date()
    #Get the fund price data
    price_data = get_price_data(ticker,start,end)
    price_data = price_data.loc[:ff_last]
    ret_data = get_return_data(price_data, "M")
    all_data = pd.merge(pd.DataFrame(ret_data),ff_data, how = 'inner', left_index= True, right_index= True)
    all_data.rename(columns={"Mkt-RF":"mkt_excess"}, inplace=True)
    all_data['port_excess'] = all_data['portfolio'] - all_data['RF']
    # Run the model
    model = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML", data = all_data).fit()
    return model


df = pd.DataFrame()

industrials_model = run_reg_model("IXM", start = "1999-05-01", end = "2020-09-30")

adj_close = get_price_data("IXM", start = "1999-05-01", end = "2020-10-31")
adj_close = adj_close.resample("M").last()
print(adj_close.tail())

#print(industrials_model.summary())
df = df.append(industrials_model.params, ignore_index=True)
print(df.head())

print("yhat =")

yhat = (df['Intercept'] + 
      (df['mkt_excess']*Rm_Rf_predict[0]) + 
      (df['SMB']*SMB_predict[0]) + 
      (df['HML']*HML_predict[0]))
      
print(yhat[0])
print(adj_close[-2])
forecast_price = (np.exp(yhat[0]))*(adj_close[-2])
print(forecast_price)

forecast_percent_error = (((adj_close[-1]) - forecast_price)/(adj_close[-1]))*100
print(forecast_percent_error)

#print(model_fit.summary())

## Ljung-Box test
#ljung_box, p_value = acorr_ljungbox(model_fit.resid)
#
#print(f'Ljung-Box test: {ljung_box[:10]}')
#print(f'p-value: {p_value[:10]}')

##plot
#X = data.values
#size = int(len(X) * 0.66)
#train, test = X[0:size], X[size:len(X)]
#history = [x for x in train]
#predictions = list()
#for t in range(len(test)):
#	model = ARIMA(history, order=(p,d,q))
#	model_fit = model.fit(disp=0)
#	output = model_fit.forecast()
#	yhat = output[0]
#	predictions.append(yhat)
#	obs = test[t]
#	history.append(obs)
#	#print('predicted=%f, expected=%f' % (yhat, obs))
##print('predicted=%f, expected=%f' % (yhat, obs))
#error = mean_squared_error(test, predictions)
#print('Test MSE: %.3f' % error)
## plot
#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()



