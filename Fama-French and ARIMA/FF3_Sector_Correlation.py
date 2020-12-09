#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:25:17 2020

@author: riku
"""

# Pandas to read csv file and other things
import pandas as pd
import yfinance as yf 
import statsmodels.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn

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
    ff_data = get_fama_french()
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


#df = pd.DataFrame()
#
#industrials_model = run_reg_model("IXI", start = "1999-05-01", end = "2019-06-30")
##print(industrials_model.summary())
#df = df.append(industrials_model.params, ignore_index=True)
#
#financials_model = run_reg_model("IXM", start = "1999-05-01", end = "2019-06-30")
##print(financials_model.summary())
#df = df.append(financials_model.params, ignore_index=True)
#
#consumer_disc_model = run_reg_model("IXY", start = "1999-05-01", end = "2019-06-30")
##print(consumer_disc_model.summary())
#df = df.append(consumer_disc_model.params, ignore_index=True)
#
#consumer_stap_model = run_reg_model("IXR", start = "1999-05-01", end = "2019-06-30")
##print(consumer_stap_model.summary())
#df = df.append(consumer_stap_model.params, ignore_index=True)
#
#com_services_model = run_reg_model("XLC", start = "1999-05-01", end = "2019-06-30")
##print(com_services_model.summary())
#df = df.append(com_services_model.params, ignore_index=True)
#
##utilities_model = run_reg_model("IXCPR", start = "1999-05-01", end = "2019-06-30")
###print(com_services_model.summary())
#
##info_tech_model = run_reg_model("IXCPR", start = "1999-05-01", end = "2019-06-30")
###print(com_services_model.summary())
#
##health_care_model = run_reg_model("IXCPR", start = "1999-05-01", end = "2019-06-30")
###print(com_services_model.summary())
#
#print(df)


#ticker = "IXI"
#startDate = '2010-01-01'
#endDate = '2020-10-30'
#
#data = yf.download(ticker, startDate, endDate)
#data.dropna(inplace=True)
#df3 = data[data['Adj Close'] > 1]  
#
##create time series plot
#df3["Adj Close"].plot()
#print(df3)
#plt.show()

symbols_list = ['IXI', 'IXM', 'IXY', 'IXR', 'XLC']

#array to store prices

df_symbols = pd.DataFrame()


#pull price using iex for each symbol in list defined above

for ticker in symbols_list: 
    start = "2000-05-01"
    end = "2018-06-30"
    r = yf.download(ticker, start, end)
    r.dropna(inplace=True)
    price_data = r[r['Adj Close'] > 1]  
    # Get FF data
    ff_data = get_fama_french()
    ff_last = ff_data.index[ff_data.shape[0] - 1].date()
    #Get the fund price data
    price_data = get_price_data(ticker,start,end)
    price_data = price_data.loc[:ff_last]
    r2 = get_return_data(price_data, "M")
    # add a symbol column
    r2['Symbol'] = ticker 
    df_symbols = df_symbols.append(r2)


# concatenate into df
df = df_symbols
print(df)
df = df.reset_index()
df = df[['Date', 'portfolio', 'Symbol']]
print(df.head())

df_pivot = df.pivot('Date','Symbol','portfolio').reset_index()
print(df_pivot)
print("mean")
print(12*df_pivot.mean())
print("stdev")
print(12*df_pivot.std())

corr_df = df_pivot.corr(method='pearson')
print(corr_df)

##reset symbol as index (rather than 0-X)
#corr_df.head().reset_index()
#del corr_df.index.name
#corr_df.head(10)
#
#print(len(corr_df))

##take the bottom triangle since it repeats itself
#
#mask = np.zeros_like(corr_df)
#
#mask[np.triu_indices_from(mask)] = True
#
##generate plot
#
##seaborn.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
#
seaborn.pairplot(df_pivot)
#
#plt.yticks(rotation=0) 
#
#plt.xticks(rotation=90) 
#
plt.show()


