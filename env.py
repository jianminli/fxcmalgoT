from conn import *
from pylab import plt
import pandas as pd
import datetime as dt

con = establish_conn()

#print(con.get_instruments())

#instruments = con.get_instruments_for_candles()
##for i in range(int(len(instruments))):
#    print(instruments[i])
#start = dt.datetime(2017, 1, 1)
#stop = dt.datetime(2018, 5, 1)

#data = con.get_candles('EUR/USD', period='H1', start=start, stop=stop)

#data.to_csv('newUSDJPYhourly.csv', sep=',')

#plt.style.use('seaborn')

#data.plot(figsize=(10,6),lw=0.8)

#con.get_open_position().T
#order = con.open_trade(symbol='USD/JPY', is_buy=True,
#                       rate=105, is_in_pips=False,
#                       amount='1000', time_in_force='GTC',
#                       order_type='AtMarket', limit=120)
#con.get_open_positions()
open_pos=con.get_open_positions()['tradeId']
print(open_pos)
con.close_trade(trade_id=open_pos[0], amount=500)
print('end')

