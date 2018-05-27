import pandas as pd
import numpy as np
#import seaborn as sns; sns.set()
from pylab import plt
plt.style.use('seaborn')
import plotly as py
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf
import datetime
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split





def data_frame():
    data = pd.read_csv('newUSDJPYhourly.csv')
    data.date = pd.to_datetime(data.date, format='%Y-%m-%d %H:%M:%S')
    data.set_index(data.date)
    data['open'] = data[['bidopen', 'askopen']].mean(axis=1)
    data['close'] = data[['bidclose', 'askclose']].mean(axis=1)
    data['high'] = data[['bidhigh', 'askhigh']].mean(axis=1)
    data['low'] = data[['bidlow', 'asklow']].mean(axis=1)
    data['volumn'] = data[['tickqty']]

    data['returns'] = np.log(data['close'] / data['close'].shift(1))
    df = data[['date', 'open', 'close', 'high', 'low', 'volumn', 'returns']]
    return df


data = data_frame()
#data.dropna(subset=['returns'])


#data[strats].dropna().cusum().apply(np.exp).plot()
#to_plot=['position_15']
#data[to_plot].iloc[-100:].plot(figsize=(10, 6), subplots=True, style=['-', '-', 'ro'], title='sb')

#data = data[-100:]
#trace1 = go.Scatter(x=data.index, y=data.position_30, name='position 30 average')

#fig = tools.make_subplots(rows=1, cols=1, shared_xaxes=True)
#fig.append_trace(trace1, 1, 1)

#py.offline.plot(fig, filename='dataprocessing.html')

#qf = cf.QuantFig(data, title='EUR/USD', datalegend=False, name='EUR/USD')
#iplot(qf.iplot(asFigure=False))

#######machine learning strategy

cols = []
lags = 8

for lag in  range(1, lags + 1):
    col = 'lag_%s' % lag
    data[col] = data['returns'].shift(lag)
    cols.append(col)

from pylab import plt
plt.style.use('seaborn')
data['direction'] = np.sign(data['returns'])
to_plot = ['close', 'returns', 'direction']
#data[to_plot].iloc[:100].plot(figsize=(10, 6), subplots=True, style=['-', '-', 'ro'], title='EUR/USD')

#patterns = 2 ** lags
#print(np.digitize(data[cols], bins=[0])[:10])
data.dropna(inplace=True)

#SVM
model = svm.SVC(C=100, probability=True) #probabality = True will take longer time
model.fit(np.sign(data[cols]), np.sign(data['returns']))
print(data.info())
print(model)

pred = model.predict(np.sign(data[cols]))
print(pred[:15])


#probabilities for market direction
pred_proba = model.predict_proba(np.sign(data[cols]))
probabilities = pd.DataFrame(pred_proba, columns=list(model.classes_))
probabilities.hist(bins=30, figsize=(8,8))
#probabilities.plot()
#plt.show()



#Vectorized backtesting
data['position'] = pred
data['strategy'] = data['position'] * data['returns']
#data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))
#plt.show()
print('position value counts')
print(data['position'].value_counts())

#Train test split
mu = data['returns'].mean()
v = data['returns'].std()
bins = [mu - v, mu, mu + v]
train_x, test_x, train_y, test_y = train_test_split(
    data[cols].apply(lambda x: np.digitize(x, bins=bins)),
    np.sign(data['returns']),
    test_size=0.50, random_state=111
)
train_x.sort_index(inplace=True)
train_y.sort_index(inplace=True)
test_x.sort_index(inplace=True)
test_y.sort_index(inplace=True)
print(train_x.head())

#model fitting & prediction
model.fit(train_x, train_y)
train_pred = model.predict(train_x)
train_acc = accuracy_score(train_y, train_pred)
test_pred = model.predict(test_x)
test_acc = accuracy_score(test_y, test_pred)
print('training accuracy = ' + str(train_acc))
print('test accuracy = ' + str(test_acc))
data.loc[test_x.index][['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))


#Vectorized backtesting - probabilities
pred_proba = model.predict_proba(np.digitize(data[cols], bins=bins))
#print('probabilities...')
#print(pred_proba[:10])

t = 0.48
pred = np.where((pred_proba[:, 0] > t) & (pred_proba[:, 1] < 0.1), -1, 0)
pred = np.where((pred_proba[:, 1] < 0.1) & (pred_proba[:, 2] > t), 1, 0)
data['position'] = pred
data['strategy'] = data['position'] * data['returns']
#in sample
data.loc[train_x.index][['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
# out of sample
data.loc[test_x.index][['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()



