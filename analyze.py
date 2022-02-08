import pandas as pd
import matplotlib.pyplot as plt

from stocks import *

columns = ['code','date','time',
           'open','high','low',
           'close','nmVolume', 'change']

filename = 'data_stock_2021-07-01_2021-12-31_1643612627.csv'
hnx = pd.read_csv(filename, usecols = columns)

codes = HNX_STOCK_LIST
list_stock_change = []
# daily_growth_threshold = 0
for code in codes:
    stock = hnx.loc[hnx['code'] == code]
    if (len(stock) == 0):
        continue
   
    data_close = stock['close']
    data_change = stock['change']
    # data_returns = data_close.pct_change()
    # data_returns = data_returns[data_returns < daily_growth_threshold]
    # data_cum_returns = (1 + data_returns).cumprod() -1

    sum_change = data_change.sum()
    change_score = data_close.head(1).item() - data_close.tail(1).item()

    # print(code, sum)
    list_stock_change.append((code, change_score))

sorted_list_change_stocks = sorted(list_stock_change,key=lambda x: x[1], reverse=True)

first_3_change = sorted_list_change_stocks[:3]
last_3_change = sorted_list_change_stocks[-3:]

print('-------------------')
print(first_3_change)
print(last_3_change)