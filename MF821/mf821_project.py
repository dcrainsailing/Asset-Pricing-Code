import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pyfinance.ols import OLS
from sklearn.linear_model import LinearRegression
import scipy
import warnings
warnings.filterwarnings('ignore')

def ad_hoc_bands_test(data, sigma, power=0.5, win_size=3, graph=True):
    Yl = theta - power*sigma
    Yu = theta + power*sigma
    tmp = data[['Price']]
    tmp['MA'] = tmp.Price.rolling(win_size).mean().dropna()
    tmp['Buy_Singal'] = tmp['MA'] + Yl
    tmp['Sell_Singal'] = tmp['MA'] + Yu
    tmp.dropna(inplace=True)
    if graph:
        plt.title(str(power)+' times sigma ' + 'and '+ str(win_size) + ' days rolling mean' )
        plt.plot(tmp.Price,label='S(t)')
        plt.plot(tmp.Buy_Singal,label='Sl(t)')
        plt.plot(tmp.Sell_Singal,label='Su(t)')
        plt.legend()
        plt.show()
    return tmp

def ad_hoc_bands_train(data, sigma, power=0.5, win_size=3, graph=True):
    Yl = theta - power*sigma
    Yu = theta + power*sigma
    tmp = data[['Price']]
    tmp['MA'] = data.Fitted_Price
    tmp['Buy_Singal'] = tmp['MA'] + Yl
    tmp['Sell_Singal'] = tmp['MA'] + Yu
    tmp.dropna(inplace=True)
    if graph:
        plt.title(str(power)+' times sigma')
        plt.plot(tmp.Price,label='S(t)')
        plt.plot(tmp.Buy_Singal,label='Sl(t)')
        plt.plot(tmp.Sell_Singal,label='Su(t)')
        plt.legend()
        plt.show()
    return tmp

def trading_strategy(data):
    df = data.copy()
    df = df.reset_index().drop(columns=['index'])
    PnL = pd.DataFrame()
    position = 0
    Size = [0]*df.shape[0]
    Value = [1000]*df.shape[0]
    i = 0
    for i in range(0,df.shape[0]):
        if position == 0:
            if df.Price[i] <= df.Buy_Singal[i]:
                position = 1
                Size[i:] = [Value[i]/df.Price[i]]*(df.shape[0]-i)
                Value[i] = Size[i]*df.Price[i]
            else:
                if i == 1:
                    Value[i] = 1000
                else:
                    Value[i] = Value[i-1]
        else:
            if df.Price[i] >= df.Sell_Singal[i]:
                position = 0
                Value[i:] = [Size[i]*df.Price[i]]*(df.shape[0]-i)
                Size[i:] = [0]*(df.shape[0]-i)
            else:
                Value[i] = Size[i]*df.Price[i]
    if Size[-1] != 0:
        Value[-1] = list(df.Price)[-1]*Size[-1]
        Size[-1] = 0
    PnL['Size'] = Size
    PnL['Value'] = Value
    PnL['Price'] = df.Price
    return PnL



if __name__ == '__main__':

    sp500 = pd.read_csv('^GSPC.csv')[['Date','Adj Close']]
    sp500.columns = ['Date','Price']
    logr = np.log(sp500['Price']/sp500['Price'].shift(1))
    logr.dropna(inplace = True)

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    ax1.hist(logr,bins=10)
    plt.title('histgram for log return of SPY')
    ax2 = fig.add_subplot(122)
    sm.qqplot(logr,fit=True,line='45',ax=ax2)
    plt.title('qqplot for log return of SPY')
    plt.show()

    train_size, test_size = 21, 21
    train, test = sp500.iloc[:train_size, :], sp500.iloc[train_size:train_size + test_size, :]

    train = train.reset_index()
    train.rename(columns={'index': 'Time'}, inplace=True)

    model = OLS(y=train.Price, x=train.Time)

    train['Fitted_Price'] = model.predicted
    alpha, beta = model.alpha, model.beta

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    sm.qqplot(model.resids, fit=True, line='45', ax=ax1)
    plt.title('qqplot for residuals after reducing linearity')

    ax2 = fig.add_subplot(122)
    plt.plot(train.Price, label='Price')
    plt.plot(train.Fitted_Price, label='fitted_price')
    plt.legend()
    plt.title('linearity of price')
    plt.show()

    data = train.copy()
    data['Y'] = data['Price'] - data['Fitted_Price']
    data['dY'] = data['Y'] - data['Y'].shift(1)
    data['dY_lag'] = data['dY'].shift(1)
    data.dropna(inplace=True)

    model = OLS(y=data.dY, x=data.dY_lag, use_const=True)
    beta_dy, alpha_dy = model.beta, model.alpha
    pd.DataFrame([[model.alpha, model.pvalue_alpha], [model.beta, model.pvalue_beta]], columns=['value', 'p-value'],
                 index=['alpha', 'beta'])

    kdt = 1 - beta_dy
    theta = np.mean(data.Y - beta_dy * data.Y.shift(1)) / (kdt)
    sigma = np.std(data.Y)
    print('theta:', round(theta, 2), 'sigma:', round(sigma, 2), 'kdt:', round(kdt, 2))

    test = test.reset_index()
    test.rename(columns={'index': 'Time'}, inplace=True)

    test['Fitted_Price'] = beta * test['Price'] + alpha

    testing = test.copy()
    testing['Y'] = testing['Price'] - testing['Fitted_Price']
    testing.dropna(inplace=True)

    result = pd.DataFrame([0, 0, 0, 0, 0, 0], index=['x_sigma', 'win', 'P&L', 'mean', 'std', 'SR'], columns=[1])
    for i in [0.5, 1.0, 1.5, 2]:
        for j in [3, 4, 5]:
            test_trading_signal = ad_hoc_bands_test(test, sigma, i, j)
            pnl = trading_strategy(test_trading_signal)
            pnl['ad_hoc_bands'] = pnl['Value'] / pnl['Value'].shift(1) - 1
            pnl['buy_and_hold'] = pnl['Price'] / pnl['Price'].shift(1) - 1
            pnl = pnl.fillna(0)
            plt.plot(np.cumprod(pnl['ad_hoc_bands'] + 1), label='ad hoc bands')
            plt.plot(np.cumprod(pnl['buy_and_hold'] + 1), label='buy_and_hold')
            plt.legend()
            plt.show()
            result[result.columns[-1] + 1] = [i, j, list(pnl.Value)[-1] / list(pnl.Value)[0] - 1, \
                                              (pnl.ad_hoc_bands.mean()) * 252, pnl.ad_hoc_bands.std() * np.sqrt(252), \
                                              (pnl.ad_hoc_bands.mean() / pnl.ad_hoc_bands.std()) * np.sqrt(252)]
    result = result.iloc[:, 1:].T
    print(result)

    test_trading_signal = ad_hoc_bands_test(test, sigma, 2, 3)
    pnl = trading_strategy(test_trading_signal)

    pnl['ad_hoc_bands'] = pnl['Value'] / pnl['Value'].shift(1) - 1
    pnl['buy_and_hold'] = pnl['Price'] / pnl['Price'].shift(1) - 1
    pnl['long_short'] = pnl['ad_hoc_bands'] - pnl['buy_and_hold']
    pnl = pnl.fillna(0)

    plt.plot(np.cumprod(pnl['ad_hoc_bands'] + 1), label='ad hoc bands')
    plt.plot(np.cumprod(pnl['buy_and_hold'] + 1), label='buy and hold')
    plt.plot(np.cumprod(pnl['long_short'] + 1), label='long short strategy')
    # plt.plot(data.Price/data.Price.iloc[0])
    plt.legend()
    plt.show()
    pd.DataFrame([[list(pnl.Value)[-1] / list(pnl.Value)[0] - 1,
                   pnl['ad_hoc_bands'].mean() * 252, pnl['ad_hoc_bands'].std() * np.sqrt(252), \
                   (pnl['ad_hoc_bands'].mean() / pnl['ad_hoc_bands'].std()) * np.sqrt(252)], \
                  [list(pnl.Price)[-1] / list(pnl.Price)[0] - 1,
                   pnl['buy_and_hold'].mean() * 252, pnl['buy_and_hold'].std() * np.sqrt(252), \
                   (pnl['buy_and_hold'].mean() / pnl['buy_and_hold'].std()) * np.sqrt(252)], \
                  [list(pnl.Value)[-1] / list(pnl.Value)[0] - list(pnl.Price)[-1] / list(pnl.Price)[0],
                   pnl['long_short'].mean() * 252, pnl['long_short'].std() * np.sqrt(252), \
                   (pnl['long_short'].mean() / pnl['long_short'].std()) * np.sqrt(252)]],
                 columns=['P&L', 'mean', 'std', 'SR'], index=['ad_bands', 'buy_hold', 'long_short'])

    k, c = kdt, 1
    Smin, Smax, M, T, N = 0, 5000, 1000, 21, 21
    ht = T / N
    ti = np.arange(0, T + ht, ht)
    hs = (Smax - Smin) / M
    Si = np.arange(Smin, Smax + hs, hs)  # j = 0,1,...,M-1

    HT = Si - c

    Ht = np.zeros((M, N))
    Ht[:, -1] = HT[:M]

    for i in range(N - 1):  # N-1):
        print(i, end=' ')
        if i % 10 == 0:
            print()
        di = 1 + (sigma ** 2) * (ht / (hs ** 2))
        li = -((sigma ** 2) / 2) * (ht / (hs ** 2)) + (
                    beta + k * (theta - Si + alpha + beta * ti[-1 - i])) * ht / 2 / hs
        ui = -((sigma ** 2) / 2) * (ht / (hs ** 2)) - (
                    beta + k * (theta - Si + alpha + beta * ti[-1 - i])) * ht / 2 / hs
        li_0 = li[1]
        ui_last = ui[-2]
        di = [di] * len(Si)

        B = np.diag(di[:M])
        x, y = B.nonzero()
        j_xy = ((x + 1)[0:-1], y[0:-1])
        u_xy = (x[0:-1], (y + 1)[0:-1])
        B[j_xy] = li[1:M]
        B[u_xy] = ui[:M - 1]

        tmp = Ht[:, -1 - i].copy()
        tmp[-1] += ui_last * (Smax - c)

        Ht[:, -1 - i - 1] = np.linalg.pinv(B).dot(tmp)

        Ht[:, -1 - i - 1] = [max(x, y) for x, y in zip(Ht[:, -1 - i - 1], Ht[:, -1])]

    Ht = pd.DataFrame(Ht)

    # Ht.head(3).append(Ht.tail(3))

    k, c = kdt, 1
    Smin, Smax, M, T, N = 0, 5000, 1000, 21, 21
    ht = T / N
    ti = np.arange(0, T + ht, ht)
    hs = (Smax - Smin) / M
    Si = np.arange(Smin, Smax + hs, hs)  # j = 0,1,...,M-1

    GT = -2 * c * np.ones(shape=Si.shape)  # Si - c

    Gt = np.zeros((M, N))
    Gt[:, -1] = GT[:M]

    for i in range(N - 1):  # N-1):
        print(i, end=' ')
        if i % 10 == 0:
            print()
        di = 1 + (sigma ** 2) * (ht / (hs ** 2))
        li = -((sigma ** 2) / 2) * (ht / (hs ** 2)) + (
                    beta + k * (theta - Si + alpha + beta * ti[-1 - i])) * ht / 2 / hs
        ui = -((sigma ** 2) / 2) * (ht / (hs ** 2)) - (
                    beta + k * (theta - Si + alpha + beta * ti[-1 - i])) * ht / 2 / hs

        li_0 = li[1]
        ui_last = ui[-2]

        di = [di] * len(Si)

        B = np.diag(di[:M])
        x, y = B.nonzero()
        j_xy = ((x + 1)[0:-1], y[0:-1])
        u_xy = (x[0:-1], (y + 1)[0:-1])
        B[j_xy] = li[1:M]
        B[u_xy] = ui[:M - 1]

        tmp = Gt[:, -1 - i].copy()
        tmp[-1] += ui_last * (-2 * c)
        #     tmp[0] += li_0 * (-c - Si[0] - c)

        Gt[:, -1 - i - 1] = np.linalg.pinv(B).dot(tmp)
        Gt[:, -1 - i - 1] = [max(x, y) for x, y in zip(Gt[:, -1 - i - 1], Ht.iloc[:, -1 - i - 1] - Si[:-1] - c)]

    Gt = pd.DataFrame(Gt)

    # Gt.head(3).append(Gt.tail(3))

    x_h = []
    for i in Ht.columns[:-2]:
        x_h.append(Ht[i][Ht[i] != Si[:-1] - c].index[-1])

    x_g = []
    for i in Gt.columns[:-2]:
        x_g.append(Gt[i][Gt[i] == Ht[i] - Si[:-1] - c].index[-1])

    plt.plot(Si[x_g], label='buy')
    plt.plot(Si[x_h], color='red', label='sell')
    plt.legend()
    plt.show()

    Su = Si[x_h]
    Sl = Si[x_g]
    Save = [alpha + beta * ti for ti in range(len(sp500))][:train_size - 2]
    Yu = Su - Save
    Yl = Sl - Save
    pp = test['Price']
    Smean = np.array(pp.rolling(3).mean().dropna())
    sell = Smean[:] + Yu[:]
    buy = Smean[:] + Yl[:]
    plt.plot(Yu, label='Yu')
    plt.plot(Yl, label='Yl')
    plt.legend()
    plt.show()

    trading_strategy_0 = pd.DataFrame([np.array(pp)[2:], buy, sell], \
                                      index=['Price', 'Buy_Singal', 'Sell_Singal']).T
    trading_strategy_0.plot()
    plt.show()

    pnl0 = trading_strategy(trading_strategy_0)

    pnl0['optimal'] = pnl0['Value'] / pnl0['Value'].shift(1) - 1
    pnl0['buy_and_hold'] = pnl0['Price'] / pnl0['Price'].shift(1) - 1
    pnl0['long_short'] = pnl0['optimal'] - pnl0['buy_and_hold']
    pnl0 = pnl0.fillna(0)

    plt.plot(np.cumprod(pnl0['optimal'] + 1), label='optimal')
    plt.plot(np.cumprod(pnl0['buy_and_hold'] + 1), label='buy and hold')
    plt.plot(np.cumprod(pnl0['long_short'] + 1), label='long short strategy')
    plt.legend()
    plt.show()

    print(pd.DataFrame([[list(pnl.Value)[-1] / list(pnl.Value)[0] - 1,
                   pnl['ad_hoc_bands'].mean() * 252, pnl['ad_hoc_bands'].std() * np.sqrt(252), \
                   (pnl['ad_hoc_bands'].mean() / pnl['ad_hoc_bands'].std()) * np.sqrt(252)], \
                  [list(pnl.Price)[-1] / list(pnl.Price)[0] - 1,
                   pnl['buy_and_hold'].mean() * 252, pnl['buy_and_hold'].std() * np.sqrt(252), \
                   (pnl['buy_and_hold'].mean() / pnl['buy_and_hold'].std()) * np.sqrt(252)], \
                  [list(pnl.Value)[-1] / list(pnl.Value)[0] - list(pnl.Price)[-1] / list(pnl.Price)[0],
                   pnl['long_short'].mean() * 252, pnl['long_short'].std() * np.sqrt(252), \
                   (pnl['long_short'].mean() / pnl['long_short'].std()) * np.sqrt(252)]],
                 columns=['P&L', 'mean', 'std', 'SR'], index=['ad_bands', 'buy_hold', 'long_short']))

    pnl_5 = pnl.copy()
    pnl_5.Size = pnl_5.Size.shift(1)
    pnl_5.Size.iloc[-1] = 0
    pnl_5 = pnl_5.iloc[1:, :]
    pnl_5.Value = pnl_5.Value.iloc[0] + (pnl_5.Size * (pnl_5.Price.shift(-1) - pnl_5.Price)).cumsum()
    pnl_5.fillna(method='ffill', inplace=True)

    pnl_5['ad_hoc_bands'] = pnl_5['Value'] / pnl_5['Value'].shift(1) - 1
    pnl_5['buy_and_hold'] = pnl_5['Price'] / pnl_5['Price'].shift(1) - 1
    pnl_5['long_short'] = pnl_5['ad_hoc_bands'] - pnl_5['buy_and_hold']
    pnl_5 = pnl_5.fillna(0)

    plt.plot(np.cumprod(pnl_5['ad_hoc_bands'] + 1), label='ad hoc bands')
    plt.plot(np.cumprod(pnl_5['buy_and_hold'] + 1), label='buy and hold')
    plt.plot(np.cumprod(pnl_5['long_short'] + 1), label='long short strategy')
    # plt.plot(data.Price/data.Price.iloc[0])
    plt.legend()
    plt.show()
    print(pd.DataFrame([[list(pnl_5.Value)[-1] / list(pnl_5.Value)[0] - 1,
                   pnl_5['ad_hoc_bands'].mean() * 252, pnl_5['ad_hoc_bands'].std() * np.sqrt(252), \
                   (pnl_5['ad_hoc_bands'].mean() / pnl_5['ad_hoc_bands'].std()) * np.sqrt(252)], \
                  [list(pnl_5.Price)[-1] / list(pnl_5.Price)[0] - 1,
                   pnl_5['buy_and_hold'].mean() * 252, pnl_5['buy_and_hold'].std() * np.sqrt(252), \
                   (pnl_5['buy_and_hold'].mean() / pnl_5['buy_and_hold'].std()) * np.sqrt(252)], \
                  [list(pnl_5.Value)[-1] / list(pnl_5.Value)[0] - list(pnl_5.Price)[-1] / list(pnl_5.Price)[0],
                   pnl_5['long_short'].mean() * 252, pnl_5['long_short'].std() * np.sqrt(252), \
                   (pnl_5['long_short'].mean() / pnl_5['long_short'].std()) * np.sqrt(252)]],
                 columns=['P&L', 'mean', 'std', 'SR'], index=['ad_bands', 'buy_hold', 'long_short']))

    pnl_5 = pnl0.copy()
    pnl_5.Size = pnl_5.Size.shift(1)
    pnl_5.Size.iloc[-1] = 0
    pnl_5 = pnl_5.iloc[1:, :]
    pnl_5.Value = pnl_5.Value.iloc[0] + (pnl_5.Size * (pnl_5.Price.shift(-1) - pnl_5.Price)).cumsum()
    pnl_5.fillna(method='ffill', inplace=True)

    pnl_5['optimal'] = pnl_5['Value'] / pnl_5['Value'].shift(1) - 1
    pnl_5['buy_and_hold'] = pnl_5['Price'] / pnl_5['Price'].shift(1) - 1
    pnl_5['long_short'] = pnl_5['optimal'] - pnl_5['buy_and_hold']
    pnl_5 = pnl_5.fillna(0)

    plt.plot(np.cumprod(pnl_5['optimal'] + 1), label='optimal bands')
    plt.plot(np.cumprod(pnl_5['buy_and_hold'] + 1), label='buy and hold')
    plt.plot(np.cumprod(pnl_5['long_short'] + 1), label='long short strategy')
    # plt.plot(data.Price/data.Price.iloc[0])
    plt.legend()
    plt.show()
    print(pd.DataFrame([[list(pnl_5.Value)[-1] / list(pnl_5.Value)[0] - 1,
                   pnl_5['optimal'].mean() * 252, pnl_5['optimal'].std() * np.sqrt(252), \
                   (pnl_5['optimal'].mean() / pnl_5['optimal'].std()) * np.sqrt(252)], \
                  [list(pnl_5.Price)[-1] / list(pnl_5.Price)[0] - 1,
                   pnl_5['buy_and_hold'].mean() * 252, pnl_5['buy_and_hold'].std() * np.sqrt(252), \
                   (pnl_5['buy_and_hold'].mean() / pnl_5['buy_and_hold'].std()) * np.sqrt(252)], \
                  [list(pnl_5.Value)[-1] / list(pnl_5.Value)[0] - list(pnl_5.Price)[-1] / list(pnl_5.Price)[0],
                   pnl_5['long_short'].mean() * 252, pnl_5['long_short'].std() * np.sqrt(252), \
                   (pnl_5['long_short'].mean() / pnl_5['long_short'].std()) * np.sqrt(252)]],
                 columns=['P&L', 'mean', 'std', 'SR'], index=['optimal_bands', 'buy_hold', 'long_short']))