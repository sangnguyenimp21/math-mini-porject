import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# from stldecompose import decompose
import pickle

# Chart drawing
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

train_filename = 'data_stock_2020-01-01_2021-11-01_1644345408.csv'
test_filename = 'data_stock_2021-07-01_2021-12-31_1643612627.csv'
columns = ['code', 'open','high','low', 'close', 'nmVolume', 'date']
drop_cols = ['date', 'nmVolume', 'open', 'low', 'high', 'code']

def relative_strength_idx(df, n=14):
    close = df['close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

if __name__ == '__main__':
    code = 'SDA'
    file_name = "xgb_"+code+".pkl"
    data = pd.read_csv(train_filename, usecols=columns)
    data = data.sort_index(axis=1 ,ascending=True)
    data = data.iloc[::-1]
    df = data.loc[data['code'] == code]

    # fig = make_subplots(rows=2, cols=1)

    # fig.add_trace(go.Ohlc(x=df.date,
    #                     open=df.open,
    #                     high=df.high,
    #                     low=df.low,
    #                     close=df.close,
    #                     name='Price'), row=1, col=1)

    # fig.add_trace(go.Scatter(x=df.date, y=df.nmVolume, name='Volume'), row=2, col=1)

    # fig.update(layout_xaxis_rangeslider_visible=False)
    # fig.show()

    df_close = df[['date', 'close']].copy()
    df_close = df_close.set_index('date')
    df_close.head()

    # decomp = decompose(df_close, period=365)
    # fig = decomp.plot()
    # fig.set_size_inches(20, 8)

    df['EMA_9'] = df['close'].ewm(9).mean().shift()
    df['SMA_5'] = df['close'].rolling(5).mean().shift()
    df['SMA_10'] = df['close'].rolling(10).mean().shift()
    df['SMA_15'] = df['close'].rolling(15).mean().shift()
    df['SMA_30'] = df['close'].rolling(30).mean().shift()

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df.date, y=df.EMA_9, name='EMA 9'))
    # fig.add_trace(go.Scatter(x=df.date, y=df.SMA_5, name='SMA 5'))
    # fig.add_trace(go.Scatter(x=df.date, y=df.SMA_10, name='SMA 10'))
    # fig.add_trace(go.Scatter(x=df.date, y=df.SMA_15, name='SMA 15'))
    # fig.add_trace(go.Scatter(x=df.date, y=df.SMA_30, name='SMA 30'))
    # fig.add_trace(go.Scatter(x=df.date, y=df.close, name='Close', opacity=0.2))
    # fig.show()

    df['RSI'] = relative_strength_idx(df).fillna(0)

    # fig = go.Figure(go.Scatter(x=df.date, y=df.RSI, name='RSI'))
    # fig.show()

    EMA_12 = pd.Series(df['close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    # fig = make_subplots(rows=2, cols=1)
    # fig.add_trace(go.Scatter(x=df.date, y=df.close, name='Close'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=df.date, y=EMA_12, name='EMA 12'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=df.date, y=EMA_26, name='EMA 26'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=df.date, y=df['MACD'], name='MACD'), row=2, col=1)
    # fig.add_trace(go.Scatter(x=df.date, y=df['MACD_signal'], name='Signal line'), row=2, col=1)
    # fig.show()

    df['close'] = df['close'].shift(-1)
    # df = df.iloc[33:] # Because of moving averages and MACD line
    df = df[:-1]      # Because of shifting close price
    df.dropna(axis=0, how='any', inplace=True)

    df.index = range(len(df))

    test_size  = 0.15
    valid_size = 0.15

    test_split_idx  = int(df.shape[0] * (1-test_size))
    valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

    train_df  = df.loc[:valid_split_idx].copy()
    valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
    test_df   = df.loc[test_split_idx+1:].copy()

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=train_df.date, y=train_df.close, name='Training'))
    # fig.add_trace(go.Scatter(x=valid_df.date, y=valid_df.close, name='Validation'))
    # fig.add_trace(go.Scatter(x=test_df.date,  y=test_df.close,  name='Test'))
    # fig.show()

    train_df = train_df.drop(drop_cols, 1)
    valid_df = valid_df.drop(drop_cols, 1)
    test_df  = test_df.drop(drop_cols, 1)

    y_train = train_df['close'].copy()
    X_train = train_df.drop(['close'], 1)

    y_valid = valid_df['close'].copy()
    X_valid = valid_df.drop(['close'], 1)

    y_test  = test_df['close'].copy()
    X_test  = test_df.drop(['close'], 1)

    # print(X_train.info())

    # parameters = {
    #     'n_estimators': [100, 200, 300, 400],
    #     'learning_rate': [0.001, 0.005, 0.01, 0.05],
    #     'max_depth': [8, 10, 12, 15],
    #     'gamma': [0.001, 0.005, 0.01, 0.02],
    #     'random_state': [42]
    # }

    parameters = {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.3],
        'max_depth': [3, 4, 5, 6],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1,2,4,8,16],
        'random_state': [42]
    }

    # save
    model = None
    if not os.path.exists(file_name):
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
        clf = GridSearchCV(model, parameters)

        clf.fit(X_train, y_train)

        print(f'Best params: {clf.best_params_}')
        print(f'Best validation score = {clf.best_score_}')

        model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        pickle.dump(model, open(file_name, "wb"))
    else:
        # load
        model = pickle.load(open(file_name, "rb"))

    # test
    # ind = 1
    # test = X_val[ind]
    # xgb_model_loaded.predict(test)[0] == model.predict(test)[0]

    # plot_importance(model)
    # plt.show()

    y_pred = model.predict(X_test)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred[:5]}')

    print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

    predicted_prices = df.loc[test_split_idx+1:].copy()
    predicted_prices['close'] = y_pred

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=df.date, y=df.close,
                            name='Truth',
                            marker_color='LightSkyBlue'), row=1, col=1)

    fig.add_trace(go.Scatter(x=predicted_prices.date,
                            y=predicted_prices.close,
                            name='Prediction',
                            marker_color='MediumPurple'), row=1, col=1)

    fig.add_trace(go.Scatter(x=predicted_prices.date,
                            y=y_test,
                            name='Truth',
                            marker_color='LightSkyBlue',
                            showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=predicted_prices.date,
                            y=y_pred,
                            name='Prediction',
                            marker_color='MediumPurple',
                            showlegend=False), row=2, col=1)

    fig.show()
