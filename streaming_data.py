from conn import *
import pandas as pd
import numpy as np

def output(data, dataframe):
    print('%3d | %s | %s, %s, %s, %s, %s'
          % (len(dataframe), data['Symbol'],
          pd.to_datetime(int(data['Updated']), unit='ms'),
          data['Rates'][0], data['Rates'][1],
          data['Rates'][2], data['Rates'][3]))

con = establish_conn()

con.subscribe_market_data('EUR/USD', (output,))

#con.get_last_price('EUR/USD')


#con.unsubscribe_market_data('EUR/USD')

#placeing orders via the restful api
#con.get_open_positions()
#order = con.create_market_buy_order('EUR/USD' , 100)
#order.get_currency()
#order.get_isBuy()
#cols = ['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy']
#con.get_open_positions()[cols]











