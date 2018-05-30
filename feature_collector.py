import numpy as np
import pandas as pd
from feature_functions import *

#load CSV data
data = pd.read_csv('USDJPYHourly.csv')

data.columns=['Date', 'open', 'high', 'low', 'close', 'BidVol']

data = data.set_index(pd.to_datetime(data.Date, format='%d.%m.%Y %H:%M:%S.%f GMT+0800'))

data = data[['open','high','low','close','BidVol']]

prices = data.drop_duplicates(keep=False)



#create list for each period
momentumKey = [3, 4, 5, 8, 9, 10]
stochasticKey = [3, 4, 5, 8, 9, 10]
williamkey = [6, 7, 8, 9, 10]
procKey = [12, 13, 14, 15]
wadlKey = [15]
cciKey = [15]
bollingerKey = [15]
heikenashiKey = [15]
paverageKey = [2]
slopeKey = [3, 4, 5, 10, 20, 30]
fourierKey = [10, 20, 30]
sineKey = [5,6]

keylist = [momentumKey, fourierKey, sineKey]

#calculate all of the features
momentumDict = momentum(prices, momentumKey)
print('1')
#hkaprices = prices.copy()
#hkaprices['Symbol'] = 'SYMBOL'
#HKA = OHLCresample(hkaprices, '15H')
#heikenashiDict = heikenAshi(HKA, heikenashiKey)
try:
    fourierDict = fourier(prices, fourierKey)
    print('2')
except:
    print('pass 2')
try:
    sineDict = sine(prices, sineKey)
    print('3')
except:
    print('pass 3')


dictlist = [momentumDict.close] #more features

#list of base column names
colFeat = ['momentum', 'fourier', 'sine']

# populate the masterframe
masterFrame = pd.DataFrame(index=prices.index)

for i in range(0, len(dictlist)):
    if colFeat[i] == 'macd':
        colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][0])
        masterFrame[colID] = dictlist[i]
    else:
        for j in keylist[i]:
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + k
                masterFrame[colID] = dictlist[i][j][k]


threshold = round(0.7 * len(masterFrame))

masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]

#heiken ashi is resampled ==> empty data in between

#masterFrame.heiken15open = masterFrame.heiken15open.fillna(method='bfill')
#high
#low
#close

#drop columns that have 30% or more nan data
masterFrameCleaned = masterFrame.copy()
masterFrameCleaned = masterFrameCleaned.dropna(axis=1, thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)

masterFrameCleaned.to_csv('Data/masterFrame.csv')
print('complete')

