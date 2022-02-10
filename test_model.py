import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

import tensorflow as tf
from tensorflow import keras

from stocks import *

pd.options.mode.chained_assignment = None

#FnB - ATS, Travel - SDA, Finance - VIG
#Drop MKV, THS, CJC
#Growth VLA, CEO, CMS
train_filename = 'data_stock_2020-01-01_2021-11-01_1644345408.csv'
test_filename = 'data_stock_2021-07-01_2021-12-31_1643612627.csv'
columns = ['code', 'open','high','low', 'close', 'nmVolume', 'date']
process_columns = [
    'close',
    'MA7', 'MA14', 'MA21',
    'HL', 'OC', 'STD7',
    'nmVolume',
    # 'EMA_9'
]
features = [
    'MA7','MA14','MA21',
    'HL', 'OC', 'STD7', 'nmVolume', 
    # 'EMA_9'
]
output_feat = ['close']

def get_csv_data(filename):
    data = pd.read_csv(filename, usecols=columns)
    data = data.sort_index(axis=1 ,ascending=True)
    data = data.iloc[::-1]

    return data

def prepare_data(list_stocks, filename):
    csv_data = get_csv_data(filename)
    data = pd.DataFrame(columns=process_columns)
    for code in list_stocks:
        stock = csv_data.loc[csv_data['code'] == code]
        if (len(stock) == 0):
            continue
        stock = stock.tail(170)
        
        stock = preprocess_data1(stock)
        data = pd.concat([data, stock])
    # data = data.drop(['index'], axis=1)
    return data

def preprocess_data1(stock):
    # stock['EMA_9'] = stock['close'].ewm(9).mean().shift()
    stock['MA7'] = stock['close'].rolling(7).mean()
    stock['MA14'] = stock['close'].rolling(14).mean()
    stock['MA21'] = stock['close'].rolling(21).mean()
    stock['HL'] = stock['high'] - stock['low']
    stock['OC'] = stock['close'] - stock['open']
    stock['STD7'] = stock['close'].rolling(7).std()
    
    stock = stock.drop([
        'code',
        'open',
        'high',
        'low'
    ], axis=1)
    # stock = stock.fillna(method ='bfill')
    stock.dropna(axis=0, how='any', inplace=True)
    return stock

def preprocess_data2(data, scaler = None, scaler_label = None):
    tempX = data.iloc[:-1]
    X = tempX[features].values

    y = data[output_feat].shift(-1)
    y.dropna(axis=0, how='any', inplace=True)
    y = y.values

    if scaler == None:
        scaler = MinMaxScaler(feature_range = (0, 1))

    if scaler_label == None:
        scaler_label = MinMaxScaler(feature_range = (0, 1))

    PredictorScalerFit=scaler.fit(X)
    TargetVarScalerFit=scaler_label.fit(y)

    X=PredictorScalerFit.transform(X)
    y=TargetVarScalerFit.transform(y)

    return X, y, scaler, scaler_label


if __name__ == '__main__':
    code_train = 'VIG'
    code_test = 'ARM'
    model_name = 'model_'+code_train+'.h5'
    data = prepare_data(list_stocks = [code_train], filename = train_filename)
    X, y, scaler, scaler_label = preprocess_data2(data = data, scaler = None, scaler_label = None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    data_test = prepare_data(list_stocks = [code_test], filename = train_filename)
    Xtest, ytest, _, _ = preprocess_data2(data = data_test, scaler = scaler, scaler_label = scaler_label)

    
    if not os.path.exists(model_name):
        model = Sequential()

        model.add(Dense(units=5, input_dim = len(features), kernel_initializer='uniform', activation='sigmoid'))
        model.add(Dense(units=5, kernel_initializer='uniform', activation='sigmoid'))

        model.add(Dense(1, kernel_initializer='uniform'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size = 16, epochs = 200)

        print(model.evaluate(X_test, y_test, verbose=1))
        print(model.summary())

        model.save(model_name)

        new_model = tf.keras.models.load_model(model_name)
    else:
        new_model = tf.keras.models.load_model(model_name)
        print(new_model.evaluate(X_test, y_test, verbose=1))
        print(new_model.summary())

    new_model = tf.keras.models.load_model(model_name)

    # results = new_model.predict(X_test)
    # converted = scaler_label.inverse_transform(results)


