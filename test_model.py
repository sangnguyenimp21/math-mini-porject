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

filename = 'data_stock_2021-07-01_2021-12-31_1643612627.csv'
columns = ['code', 'open','high','low', 'close', 'nmVolume']
process_columns = ['close', 'MA7','MA14','MA21', 'HL', 'OC', 'STD7', 'nmVolume']
features = ['MA7','MA14','MA21', 'HL', 'OC', 'STD7', 'nmVolume']
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

        stock = preprocess_data1(stock)
        data = pd.concat([data, stock])
    # data = data.drop(['index'], axis=1)
    return data

def preprocess_data1(stock):
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
    tempX = data.iloc[:-7]
    X=tempX[features].values
    
    y = data[output_feat].shift(-7)
    y.dropna(axis=0, how='any', inplace=True)
    y= y.values

    if scaler == None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    if scaler_label == None:
        scaler_label = MinMaxScaler(feature_range=(0, 1))
    
    PredictorScalerFit=scaler.fit(X)
    TargetVarScalerFit=scaler_label.fit(y)

    X=PredictorScalerFit.transform(X)
    y=TargetVarScalerFit.transform(y)

    return X, y, scaler, scaler_label

if __name__ == '__main__':
    data = prepare_data(list_stocks = ['ATS'], filename = filename)
    X, y, scaler, scaler_label = preprocess_data2(data = data, scaler=None, scaler_label = None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    data_test = prepare_data(list_stocks = ['ARM'], filename = filename)
    Xtest, ytest, _, _ = preprocess_data2(data = data_test, scaler=scaler, scaler_label = scaler_label)


    if not os.path.exists('my_model.h5'):
        model = Sequential()

        model.add(Dense(units=5, input_dim=len(features), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=5, kernel_initializer='uniform', activation='tanh'))
        model.add(Dense(1, kernel_initializer='uniform'))
        
        # model.add(Dense(3, kernel_initializer='uniform', activation='relu', input_shape=(len(features), 1)))
        # model.add(Dense(1, kernel_initializer='normal'))

        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=100)

        print(model.evaluate(X_test, y_test, verbose=1))
        print(model.summary())

        model.save('my_model.h5')

        new_model = tf.keras.models.load_model('my_model.h5')
    else:
        new_model = tf.keras.models.load_model('my_model.h5')
        print(new_model.evaluate(X_test, y_test, verbose=1))
        print(new_model.summary())

    new_model = tf.keras.models.load_model('my_model.h5')
    results = new_model.predict(Xtest)
    # print(results)
    a = scaler_label.inverse_transform(results)
    print(a)
    b = scaler_label.inverse_transform(y_test)
    print(b)
    
    


