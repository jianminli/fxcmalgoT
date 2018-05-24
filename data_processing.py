import pandas as pd
import numpy as np


def data_frame():
    data = pd.read_csv('newUSDJPYhourly.csv')
    data.date = pd.to_datetime(data.date, format='%Y-%m-%d %H:%M:%S')
    data.set_index(data.date)
    data['open'] = data[['bidopen', 'askopen']].mean(axis=1)
    data['close'] = data[['bidclose', 'askclose']].mean(axis=1)
    data['high'] = data[['bidhigh', 'askhigh']].mean(axis=1)
    data['low'] = data[['bidlow', 'asklow']].mean(axis=1)

    df = data[['date', 'open', 'close', 'high', 'low', 'tickqty']]

    return df


data = data_frame()

data['returns'] = np.log(data['close'] / data['close'].shift(1))

cols = []

for momentum in [15, 20, 30]:  # 14
    col = 'position_%s' % momentum  # 15
    data[col] = np.sign(data['returns'].rolling(momentum).mean())  # 16
    cols.append(col)  # 17

print(data.head(100))

