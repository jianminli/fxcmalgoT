import numpy as np
import pandas as pd
from feature_functions import *

#load CSV data
data = pd.read_csv('USDJPYHourly.csv')

data.columns=['Date', 'open', 'high', 'low', 'close', 'BidVol']

data = data.set_index(pd.to_datetime(data.Date, format='%d.%m.%Y %H:%M:%S.%f GMT+0800'))

data = data[['open','high','low','close','BidVol']]

prices = data.drop_duplicates(keep=False)

print(data.head())




#create list for each period


#calculate all of the features
