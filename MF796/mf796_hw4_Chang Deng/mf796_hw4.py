import numpy as np
import json
from pandas_datareader import data as pdr
import yfinance as yf
from cvxopt import matrix, solvers
from pyfinance.ols import OLS
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

def eigenvalue_decomposition(cov_martix):
    eigenvalue,eigenvector = np.linalg.eig(cov_martix)
    return eigenvalue,eigenvector

def eigenvalue_for_variance(eigVals,percentage):
    descending = -np.sort(-eigVals)
    totalSum = sum(eigVals)
    localSum = 0
    num = 0
    for i in descending:
        localSum += i
        num += 1
        if localSum >= totalSum * percentage:
            return num

def pca(dataMat,n):
    meanVal = 0
    newData = dataMat
    #meanVal=np.mean(dataMat,axis=0)
    #newData=dataMat-meanVal
    covMat=np.cov(dataMat,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    HU = newData*n_eigVect
    #HUH = (HU*eigVects.T)+meanVal
    return HU

def portfolio_optimaztion(R, C, a=1):
    row,col = C.shape
    Q = a*(2*matrix(C))
    p = matrix(-1*R)
    G = matrix(-1*np.identity(col))
    h = matrix(np.zeros(shape=(col,1),dtype=float))
    A = np.zeros(shape=(2,col),dtype=float)
    A[0] = 1
    A[1][0:17] = 1
    A = matrix(A)
    b = np.zeros(shape=(2,1),dtype=float)
    b[0] = 1
    b[1] = 0.1
    b = matrix(b)
    Lambda = np.dot(np.linalg.inv(np.matrix(A)*np.linalg.inv(np.matrix(C))*np.matrix(A).T),-(2*a*b - np.matrix(A)*np.linalg.inv(np.matrix(C))*np.matrix(R).T))
    w = (np.linalg.inv(np.matrix(C))*(np.reshape(R,(R.shape[0],1)) - np.matrix(A).T*Lambda))/(2*a)
    sol = solvers.qp(P=Q,q=p,G=G,h=h,A=A,b=b)
    w1 = sol['x']
    return w,w1


if __name__ == '__main__':
    # display all columns
    pd.set_option('display.max_columns', None)
    # display all rows
    pd.set_option('display.max_rows', None)
    # problem 1
    # 1.1
    sp500 = pdr.get_data_yahoo('^GSPC', start='2015-02-12', end='2020-02-12')
    data = sp500.reset_index().loc[:, ['Date']]
    with open("sp500-historical-components.json", 'r') as load_f:
        load_dict = json.load(load_f)
    stk_list = load_dict[1]['Symbols']
    stk_list = stk_list[0:110]
    for ticker in stk_list:
        try:
            stk = pdr.get_data_yahoo(ticker, start='2015-02-12', end='2020-02-12')
        except:
            # print('no ticker: ', ticker)
            stk_list.remove(ticker)
        else:
            if stk.empty != True:
                stk = stk.reset_index().loc[:, ['Date', 'Close']]
                stk.rename(columns={'Close': ticker + '_Close'}, inplace=True)
                data = pd.DataFrame.merge(data, stk, how='left', on='Date')
    data.dropna(axis=1, how='all', inplace=True)
    values = dict([(col_name, col_mean) for col_name, col_mean in zip(data.columns.tolist(), data.mean().tolist())])
    data.fillna(value=values, inplace=True)
    # 1.2
    for columns in data.columns.tolist()[1:]:
        data[columns[:-6] + '_daily_return'] = np.log(data[columns] / data[columns].shift(1))
        data.drop(columns=columns, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    #print(data.isnull().any())
    #1.3
    daily_return = data.drop(columns='Date')
    corr = daily_return.corr()
    cov = daily_return.cov()
    eigenvalue,eigenvector = eigenvalue_decomposition(cov.values)
    plt.plot(-np.sort(-eigenvalue))
    plt.axhline(y=0, c="red")
    plt.show()
    #1.4
    p50 = eigenvalue_for_variance(eigenvalue,0.5)
    #print(p50)
    p90 = eigenvalue_for_variance(eigenvalue,0.9)
    #print(p90)
    #1.5
    a = pca(np.array(daily_return.values), p90)
    re = np.sum(daily_return.values, axis=1)
    beta = np.linalg.inv(np.matrix(a.T) * np.matrix(a)) * np.matrix(a.T) * (np.matrix(re).T)
    ee = (np.matrix(re).T) - np.matrix(a) * np.matrix(beta)
    plt.plot(ee)
    plt.show()
    sm.qqplot(np.array(ee.flatten())[0], fit=True, line='45')
    plt.show()
    plt.figure(figsize=(35, 35))
    i = 0
    while i < daily_return.shape[1]:
        r = daily_return.iloc[:, i].values
        beta = np.linalg.inv(np.matrix(a.T) * np.matrix(a)) * np.matrix(a.T) * (np.matrix(r).T)
        ee = (np.matrix(r).T) - np.matrix(a) * np.matrix(beta)
        plt.plot(ee, label=daily_return.columns[i])
        i += 1
    plt.legend()
    plt.show()
    # problem 2
    annual_return = 252 * daily_return.mean().values
    annual_cov = 252 * cov.values
    w, w1 = portfolio_optimaztion(annual_return, annual_cov)
    plt.plot(w)
    plt.show()
    plt.plot(w1)
    plt.show()