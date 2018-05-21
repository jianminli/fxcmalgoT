import pandas as pd
import plotly as py
import matplotlib.pyplot as plt
from plotly import tools
import plotly.graph_objs as go
from feature_functions import *

df = pd.read_csv('newUSDJPYhourly.csv')
#date,bidopen,bidclose,bidhigh,bidlow,askopen,askclose,askhigh,asklow,tickqty
#2017-01-03 00:00:00
df.date = pd.to_datetime(df.date, format='%Y-%m-%d %H:%M:%S')
df.set_index(df.date)
df.dropna()
#data = df[['date', 'bidopen']]
df = df.tail(500)

df_bid = df[['date', 'bidopen', 'bidclose', 'bidhigh', 'bidlow']]
df_bid.columns = ['date', 'open', 'close', 'high', 'low']
#print(df_bid['close'] / df_bid['close'].shift(1))
df_bid['returns'] = np.log(df_bid['close'] / df_bid['close'].shift(1))
#temp
#df_bid = df_bid.tail(100)

#momentum
m = momentum(df_bid, [15, 20])
#print(m.open[20])
res = m.close[20]



detrended_close = detrend(df_bid, method='difference')
print(detrended_close)


#plotting
bid_trace = go.Ohlc(x=df.date, open=df.bidopen, high=df.bidhigh, low=df.bidlow, close=df.bidclose, name='EUR USD bid hourly')
ask_trace = go.Ohlc(x=df.date, open=df.askopen, high=df.askhigh, low=df.asklow, close=df.askclose, name='EUR USD ask hourly')
detrended_trace = go.Scatter(x=df.date, y=detrended_close, name='EUR USD bid detrended close')
momentum_trace = go.Scatter(x=df.date, y=res.close, name='momentum')

#data = [bid_trace]

fig = tools.make_subplots(rows=4, cols=1, shared_xaxes=True)
fig.append_trace(bid_trace, 1, 1)
fig.append_trace(ask_trace, 2, 1)
fig.append_trace(momentum_trace, 3, 1)
fig.append_trace(detrended_trace, 4, 1)

py.offline.plot(fig, filename='plotting2.html')

