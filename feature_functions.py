
import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.finance import _candlestick
from matplotlib.dates import date2num
from datetime import datetime

class holder:
    1

def heikenAshi(prices,periods):
    """
    #Heiken Ashi candles

    #HA(close) = O + H + L + C /4
    #HA (Open) = HA(open,prev)+HA(close,prev)/2
    #HA (high) = max(high,HAopen, HAclose)
    #HA (low) = min (low,HAopen,HAclose)

    :param prices: dataframe of OHLC & volumn
    :param periods: periods for which to create the candles
    :return: Heiken Ashi OHLC candles

    """
    results = holder()
    dict = {} #"store the candles"
    HAclose = prices[['open', 'high', 'low', 'close']].sum(axis=1)/4
    HAopen = HAclose.copy()
    HAopen.iloc[0] = HAclose.iloc[0]
    HAhigh = HAclose.copy()
    HAlow = HAclose.copy()

    for i in range(1, len(prices)):
        HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
        HAhigh.iloc[i] = np.array([prices.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        HAlow.iloc[i] = np.array([prices.low.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).min()

    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.columns = [['open', 'high', 'low', 'close']]

    #df.index=df.index.droplevel(0)

    dict[periods[0]] = df

    results.candles = dict

    return results


#Detrender
def detrend(prices, method='difference'):

    """

    :param prices: dataframe of OHLC currency data
    :param method: method by which to detrend "liner" or "difference"
    :return: the detrended price series
    """

    if method == 'difference':
        detrend = prices.close[1:] - prices.close[:-1].values
    elif method == 'linear':
        x = np.arange(0, len(prices))
        y = prices.close.values

        model = LinearRegression()

        model.fit(x.reshape(-1,1), y.reshape(-1,1))

        trend = model.predict(x.reshape(-1,1))

        trend = trend.reshape((len(prices),))

        detrend = prices.close - trend

    else:
        print('you didnot input a valid method')

    return detrend

# Fourier series expansion fitting function
#F=a0+a1cos(wx)+b1sin(wx)
def fseries(x,a0,a1,b1,w):
    """

    :param x: The hours
    :param a0:
    :param a1:
    :param b1:
    :param w: Fourier series frenquence
    :return:
    """

    f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)
    return f


# Sine series expansion fitting function
#F=a0+b1sin(wx)
def sseries(x,a0,a1,b1,w):
    """

    :param x: The hours
    :param a0:
    :param a1:
    :param b1:
    :param w: Fourier series frenquence
    :return:
    """

    f = a0 + b1*np.sin(w*x)
    return f


#Fourier series coefficient calculator function
def fourier(prices,periods,method='difference'):
    """

    :param prices: OHLC dataframe
    :param periods: list of periods for which to compute coefficients [3,5,10///]
    :param method: method by which to detrend the data
    :return: dict of dataframes containing coefficients for said periods
    """
    results = holder()
    dict = {}
    #Option to plot the expansion fit for each iteration
    plot = True
    #Compute the coefficients of the series
    detrended = detrend(prices,method)

    for i in range(0,len(periods)):
        coeffs = []
        for j in range(periods[i],len(prices)-periods[i]):
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
                try:
                    res = scipy.optimize.curve_fit(fseries,x,y)
                except(RuntimeError, OptimizeWarning):
                    res = np.empty((1,4))
                    res[0,:]=np.NAN
            if plot == True:
                xt = np.linspace(0,periods[i],100)
                yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                plt.show()
            coeffs = np.append(coeffs,res[0],axis=0)

        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        coeffs = np.array(coeffs).reshape(((len(coeffs)/4,4)))
        df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]])
        df.columns = [['a0','a1','b1','w']]
        df = df.fillna(method='bfill')
        dict[periods[i]] = df
    results.coeffs = dict
    return results


#Sine series coefficient calculator function
def sine(prices,periods,method='difference'):
    """

    :param prices: OHLC dataframe
    :param periods: list of periods for which to compute coefficients [3,5,10///]
    :param method: method by which to detrend the data
    :return: dict of dataframes containing coefficients for said periods
    """
    results = holder()
    dict = {}
    #Option to plot the expansion fit for each iteration
    plot = True
    #Compute the coefficients of the series
    detrended = detrend(prices,method)

    for i in range(0,len(periods)):
        coeffs = []
        for j in range(periods[i],len(prices)-periods[i]):
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
                try:
                    res = scipy.optimize.curve_fit(sseries,x,y)
                except(RuntimeError, OptimizeWarning):
                    res = np.empty((1,3))
                    res[0,:]=np.NAN
            if plot == True:
                xt = np.linspace(0,periods[i],100)
                yt = sseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                plt.show()
            coeffs = np.append(coeffs,res[0],axis=0)

        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        coeffs = np.array(coeffs).reshape(((len(coeffs)/3,3)))
        df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]])
        df.columns = [['a0','a1','b1','w']]
        df = df.fillna(method='bfill')
        dict[periods[i]] = df
    results.coeffs = dict
    return results


#William accumulation distribution
def wadl(prices,periods):
    """

    :param prices: dataframe of OHLC prices
    :param periods: (list) periods for which to calculate the function [5,10,15...]
    :return: williams accumulation distribution lines for each period
    """
    results = holder()
    dict = {}
    for i in range(0, len(periods)):
        WAD = []
        for j in range(periods[i], len(prices)-periods[i]):
            TRH = np.array([prices.high.iloc[j], prices.close.iloc[j - 1]]).max()
            TRL = np.array([prices.low.iloc[j], prices.close.iloc[j - 1]]).min()

            if prices.close.iloc[j] > prices.close.iloc[j-1]:
                PM = prices.close.iloc[j] - TRL
            elif prices.close.iloc[j] <  prices.close.iloc[j-1]:
                PM = prices.close.iloc[j] - TRH
            elif prices.close.iloc[j] == prices.close.iloc[j-1]:
                PM = 0
            else:
                print('Unknown error occured')
            AD = PM*prices.volume.iloc[j]

            WAD = np.append(WAD, AD)

        WAD = WAD.cumsum()

        WAD = pd.DataFrame(WAD, index=prices.iloc[periods[i]:-periods[i]].index)

        WAD.columns = [['close']]

        dict[periods[i]] = WAD

    results.wadl = dict

    return results


def momentum(prices,periods):
    """

    :param prices: data frame of OHLC
    :param periods: list of periods to calculate functions
    :return: momentum indicator
    """
    results = holder()
    open = {}
    close = {}
    for i in range(0, len(periods)):
        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)
        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)
        #open[periods[i]].columns=[['open']]
        #close[periods[i]].columns=[['close']]

    results.open=open
    results.close=close

    return results

