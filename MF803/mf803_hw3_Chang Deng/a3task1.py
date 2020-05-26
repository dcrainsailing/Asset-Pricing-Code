from a1task2_803 import *
import yfinance as yf
from pandas_datareader import data as pdr
from pandas.plotting import register_matplotlib_converters
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
import statsmodels.api
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats.stats import pearsonr
import datetime
import numpy as np


if __name__ == '__main__':
    register_matplotlib_converters()
    # print(plt.style.available)
    plt.style.use('ggplot')
    # display all columns
    pd.set_option('display.max_columns', None)
    # display all rows
    pd.set_option('display.max_rows', None)

    yf.pdr_override()
    vix = pdr.get_data_yahoo('^VIX', start='1993-01-29', end='2019-09-24')
    vix = vix.reset_index()
    vix = vix.loc[:, ['Date', 'Close']]
    vix.rename(columns={'Close': 'Volatility_proxy'}, inplace=True)
    spy = pdr.get_data_yahoo('SPY', start='1993-01-29', end='2019-09-24')
    spy = spy.reset_index()
    spy = spy.loc[:, ['Date', 'Close']]
    # spy['ETF_Daily_return'] = spy['Close'] / spy['Close'].shift(1) - 1
    # spy = spy.fillna(0)
    # spy = spy.drop(columns=['Close'])
    print(vix.describe())
    print(spy.describe())
    #question b
    plot_acf(vix['Volatility_proxy'])
    model = statsmodels.tsa.stattools.acf(vix['Volatility_proxy'], unbiased=False, nlags=1, qstat=True,
                                          alpha=None, fft=False, missing='none')
    print(model)
    plot_acf(spy['Close'])
    model = statsmodels.tsa.stattools.acf(spy['Close'], unbiased=False, nlags=1, qstat=True,
                                          alpha=None, fft=False, missing='none')
    print(model)
    plt.show()
    # question c
    data = pd.DataFrame.merge(spy, vix, how='left', on='Date')
    date = data['Date']
    corr = pearsonr(data['Close'], data['Volatility_proxy'])
    print(corr)
    monthly_data = (data.set_index('Date')).resample('M', convention='end').ffill()
    # data['Month'] = (date.apply(lambda x: datetime.datetime.strftime(x,"%Y-%m-%d"))).apply(lambda x: x[0:7])
    # monthly_data = (data.groupby(['Month']).agg("sum", axis="columns")).reset_index()
    monthly_corr = pearsonr(monthly_data['Close'], monthly_data['Volatility_proxy'])
    print(monthly_corr)
    # question d
    rolling_corr = data['Close'].rolling(90).corr(data['Volatility_proxy'])
    rolling_corr = (rolling_corr.dropna(axis=0, how='any')).reset_index(drop=True)
    X = (data.loc[89:data.shape[0], 'Date']).reset_index(drop=True)
    plt.plot(X, rolling_corr)
    plt.axhline(y=np.mean(rolling_corr), c="yellow")  
    plt.axvline(x=data.loc[89 + np.argmax(np.array(rolling_corr)), 'Date'], c="blue", linestyle=":")
    print(data.loc[89 + np.argmax(np.array(rolling_corr)), 'Date'])
    plt.show()
    # question e
    data['ETF_Daily_return'] = data['Close'] / data['Close'].shift(1)
    data['ETF_Daily_return'] = np.square(np.log(data.loc[:, 'ETF_Daily_return']))
    data = data.fillna(0)
    realized_vol = 100 * np.sqrt(252 * data['ETF_Daily_return'].rolling(90).sum() / 90)
    realized_vol = (realized_vol.dropna(axis=0, how='any')).reset_index(drop=True)
    premium = pd.DataFrame()
    premium['Date'] = ((data.loc[89:data.shape[0], 'Date']).copy()).reset_index(drop=True)
    premium['Implied_volatility'] = ((data.loc[89:data.shape[0], 'Volatility_proxy']).copy()).reset_index(drop=True)
    premium['Realized_volatility'] = realized_vol.copy()
    premium['Premium'] = premium['Implied_volatility'] - premium['Realized_volatility']
    plt.plot((data.loc[89:data.shape[0], 'Date']), premium['Premium'])
    plt.axhline(y=0, c="yellow")
    plt.show()
    #print(premium.loc[np.argmax(np.array(premium['Premium']), axis=0), 'Date'])
    #print(premium.loc[np.argmin(np.array(premium['Premium']), axis=0), 'Date'])
    # question f
    option_price = []
    payoffs = []
    spy['Close_lead'] = data['Close'].shift(-20)
    i = 0
    while i < spy.shape[0]:
        S0 = spy.loc[i, 'Close']
        vol = vix.loc[i, 'Volatility_proxy']
        euro_op = european_option(S0=S0, sigma=vol / 100, K=S0, T=1 / 12)
        option_price = option_price + [euro_op.formulaic_price('put') + euro_op.formulaic_price('call')]
        payoffs = payoffs + [abs(spy.loc[i, 'Close'] - spy.loc[i, 'Close_lead'])]
        i = i + 1
    result = {'Date': spy['Date'], 'Option_price': option_price, 'Payoffs': payoffs}
    result = pd.DataFrame(result)
    result = result.fillna(0)
    # question g
    result['Daily_P&L'] = result['Payoffs'] - result['Option_price']
    result = (result.dropna()).reset_index(drop=True)
    # result.loc[0,'Accumulative_S&P'] = result.loc[0,'Payoffs'] - result.loc[0,'Option_price']
    # j = 1
    # while j < spy.shape[0]:
    #     result.loc[j, 'Accumulative_S&P'] = result.loc[j-1, 'Accumulative_S&P'] + result.loc[j,'Payoffs'] - result.loc[j,'Option_price']
    #     j = j + 1
    #plt.plot(result['Date'], -result['Daily_P&L'])
    #plt.plot(result['Date'], result['Payoffs'])
    #plt.plot(result['Date'], result['Accumulative_S&P'])
    #plt.show()
    #print(np.mean(result['Option_price']))
    #print(np.mean(result['Payoffs']))
    plt.plot(result['Date'], result['Daily_P&L'])
    plt.axhline(y=0, c="yellow")
    plt.show()
    print(np.mean(result['Daily_P&L']))
    #question h
    contrast = pd.DataFrame.merge(premium, result, how='left', on='Date')
    contrast = (contrast.dropna()).reset_index(drop=True)
    plt.plot(contrast['Date'], contrast['Premium'])
    plt.plot(contrast['Date'], contrast['Daily_P&L'])
    plt.show()
    contrast.plot.scatter(x='Premium', y='Daily_P&L')
    # calc the trendline
    z = np.polyfit(contrast['Premium'], contrast['Daily_P&L'], 1)
    p = np.poly1d(z)
    plt.plot(contrast['Premium'], p(contrast['Premium']), "r--")
    # the line equation:
    plt.title("y=%.6fx+(%.6f)" % (z[0], z[1]))
    plt.show()
