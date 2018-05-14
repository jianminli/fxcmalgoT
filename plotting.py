import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go
from feature_functions import *

df = pd.read_csv('USDJPYHourly.csv')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df.date = pd.to_datetime(df.date, format='%d.%m.%Y %H:%M:%S.%f GMT+0800')
df.set_index(df.date)
df = df[['open', 'high', 'low', 'close', 'volume']]
df = df.drop_duplicates(keep=False)
#df = df.iloc[:200]

ma = df.close.rolling(center=False, window=30).mean() #move average

# 2. get function data from selected function
HAresults = heikenAshi(df, [1]) # 1hour period
HA = HAresults.candles[1]

detrended = detrend(df, method='difference')

#f = fourier(df,[10,15],method='difference')
WADl = wadl(df, [15])
line = WADl.wadl[15]

# 3. plot

trace0 = go.Ohlc(x=df.index, open=df.open,high=df.high,low=df.low,close=df.close,name='USDJPY')
trace1 = go.Scatter(x=df.index, y=ma)
#trace2 = go.Bar(x=df.index, y=df.volume)
trace2 = go.Ohlc(x=HA.index, open=HA.open, high=HA.high, low=HA.low, close=HA.close, name='Heiken Ashi')
trace3 = go.Scatter(x=df.index, y=detrended)
trace4 = go.Scatter(x=line.index, y=line.close)
data = [trace0, trace1, trace2, trace3]

fig = tools.make_subplots(rows=4, cols=1, shared_xaxes=True)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)
fig.append_trace(trace3,3,1)

py.offline.plot(fig, filename='plotting.html')
