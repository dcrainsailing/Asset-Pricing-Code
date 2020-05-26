import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
from a1task1_803 import *
from cvxopt import matrix, solvers

def annualized_return(code_list=['SPY'],start_date='2010-01-01',end_date='2019-11-18'):
    annual_return = []
    for code in code_list:
        etf = ETF(code,start_date,end_date)
        etf.price_acquire()
        data = etf.data.reset_index()
        #print(data['Close'].describe())
        annually_data = ((data.set_index('Date')).resample('Y', convention='end').ffill()).reset_index()
        initial_begin = data.loc[0, 'Close']
        initial_end = annually_data.loc[0, 'Close']
        initial_return = initial_end / initial_begin
        annually_data['Gaps'] = annually_data['index'] - annually_data['index'].shift(1).fillna(0)
        log_return = (annually_data['Close'] / annually_data['Close'].shift(1)).fillna(initial_return)
        annual_return = annual_return + [np.mean(log_return.apply(math.log) * 252 / annually_data['Gaps'])]
    ann = pd.DataFrame(index=code_list,columns=['Annualized_Return'],data=annual_return)
    return ann

def covariance_matrix(code_list=['SPY'],start_date = '2010-01-01',end_date = '2019-11-18'):
    r = pd.DataFrame()
    spy = ETF('SPY',start_date, end_date)
    spy.price_acquire()
    r['Date'] = spy.data['Date']
    for code in code_list:
        etf = ETF(code,start_date, end_date)
        etf.price_acquire()
        data = etf.data
        data['' + code + '_daily_return'] = data['Close'] / data['Close'].shift(1) - 1
        data = (data.dropna(axis=0, how='any'))
        data = data.drop(columns='Close')
        r = pd.DataFrame.merge(r,data,how='left',on='Date')
    r = r.dropna(axis=0, how='any')
    r = r.drop(columns='Date')
    corr = r.corr()
    cov = r.cov()
    return corr,cov

def eigenvalue_decomposition(cov_martix):
    e, q = np.linalg.eig(cov_martix)
    e = -np.sort(-e)
    return e

def portfolio_optimaztion(R, C, a=1):
    # C = (C.copy()).values
    # row,col = C.shape
    # Q = a*(2*matrix(C))
    # p = matrix(-1*R)
    # G = matrix(-1*np.identity(col))
    # h = matrix(np.zeros(shape=(col,1),dtype=float))
    # A = matrix(np.ones(shape=(1,col),dtype=float))
    # b = matrix(np.ones(shape=(1,1),dtype=float))
    # sol = solvers.qp(Q, p)
    # w = sol['x']
    #print(sol['primal objective'])
    C_ = np.linalg.inv(C)
    w = (1 / 2) * (matrix(C_) * matrix(R))
    return w

def expected_return(R,sigma):
    mat_shape = R.shape
    z = np.random.normal(0.0, 1.0, mat_shape)
    Er = R + sigma*z
    return Er

def regularized_covariance(cov,delta):
    full = matrix((cov.copy()).values)
    diag = matrix(np.diagflat(np.diag(cov)))
    rcov = delta*diag + (1-delta)*full
    return rcov

if __name__ == '__main__':
    # display all columns
    pd.set_option('display.max_columns', None)
    # display all rows
    pd.set_option('display.max_rows', None)
    #problem 1
    #b
    cor, cov = covariance_matrix(['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'])
    cov = 252*cov.copy()
    #c
    e = eigenvalue_decomposition((cov).values)
    print(e)
    plt.plot(e)
    plt.axhline(y=0, c="red")
    plt.show()
    #d
    mat_shape = (cov.values).shape
    norm_rnds = np.random.normal(0.0, 1.0, mat_shape)
    #e
    e2 = eigenvalue_decomposition(norm_rnds)
    plt.plot(np.real(e2))
    plt.axhline(y=0, c="red")
    plt.show()
    plt.plot(np.imag(e2))
    plt.axhline(y=0, c="red")
    plt.show()
    #problem 2
    #a
    ann_return = annualized_return(['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'])
    ann_return = (ann_return.copy()).values
    w1 = portfolio_optimaztion(ann_return,cov)
    print(w1)
    #b、c
    adj_returns = expected_return(ann_return,0.005)
    w2 = portfolio_optimaztion(adj_returns,cov)
    print((np.abs(w2-w1)).sum())
    adj_returns = expected_return(ann_return, 0.01)
    w3 = portfolio_optimaztion(adj_returns, cov)
    print(np.abs(w3-w1).sum())
    adj_returns = expected_return(ann_return, 0.05)
    w4 = portfolio_optimaztion(adj_returns, cov)
    print(np.abs(w4-w1).sum())
    adj_returns = expected_return(ann_return, 0.1)
    w5 = portfolio_optimaztion(adj_returns, cov)
    print(np.abs(w5-w1).sum())
    #d、e
    reg_cov1 = regularized_covariance(cov, 1)
    print(np.linalg.matrix_rank(reg_cov1))
    e1 = eigenvalue_decomposition((reg_cov1))
    print(e1)
    #f
    reg_cov2 = regularized_covariance(cov, 0.1)
    print(np.linalg.matrix_rank(reg_cov2))
    e2 = eigenvalue_decomposition((reg_cov2))
    print(e2)
    reg_cov3 = regularized_covariance(cov, 0.2)
    print(np.linalg.matrix_rank(reg_cov3))
    e3 = eigenvalue_decomposition((reg_cov3))
    print(e3)
    reg_cov4 = regularized_covariance(cov, 0.5)
    print(np.linalg.matrix_rank(reg_cov4))
    e4 = eigenvalue_decomposition((reg_cov4))
    print(e4)
    reg_cov5 = regularized_covariance(cov, 0.7)
    print(np.linalg.matrix_rank(reg_cov5))
    e5 = eigenvalue_decomposition((reg_cov5))
    print(e5)
    #g
    w6 = portfolio_optimaztion(adj_returns, reg_cov2)
    print(np.abs(w6 - w5).sum())
    w7 = portfolio_optimaztion(adj_returns, reg_cov3)
    print(np.abs(w7 - w5).sum())
    w8 = portfolio_optimaztion(adj_returns, reg_cov4)
    print(np.abs(w8 - w5).sum())
    w9 = portfolio_optimaztion(adj_returns, reg_cov5)
    print(np.abs(w9 - w5).sum())