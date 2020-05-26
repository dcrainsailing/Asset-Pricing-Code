import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve
from scipy.optimize import minimize
from pyfinance import ols
import matplotlib.pyplot as plt

def call_delta(K,sigma,T,S0=100,r=0):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) \
             / (sigma * np.sqrt(T))
    return stats.norm.cdf(d1)

def put_delta(K,sigma,T,S0=100,r=0):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) \
             / (sigma * np.sqrt(T))
    return stats.norm.cdf(-d1)

def call_price(K_list,vol_list,T,S0=100,r=0):
    i = 0
    c_list = []
    while i<len(K_list):
        d1 = (np.log(S0 / K_list[i]) + (r + 0.5 * vol_list[i] ** 2) * T) / (vol_list[i] * np.sqrt(T))
        d2 = d1 - vol_list[i] * np.sqrt(T)
        c_list.append(S0 * stats.norm.cdf(d1) - K_list[i] * np.exp(-r * T) * stats.norm.cdf(d2))
        i += 1
    return c_list

def digital_put(K,K_list,d_list,w):
    i = 0
    price = 0
    while i < len(d_list):
        if K_list[i] > K:
            price += 0
        else:
            price += 1*d_list[i]*w
        i += 1
    return price

def digital_call(K,K_list,d_list,w):
    i = 0
    price = 0
    while i < len(d_list):
        if K_list[i] < K:
            price += 0
        else:
            price += 1*d_list[i]*w
        i += 1
    return price

def call(K,K_list,d_list,w):
    i = 0
    price = 0
    while i < len(d_list):
        price += np.maximum(K_list[i]-K,0)*d_list[i]*w
        i += 1
    return price

def characteristic_function(u,t,sigma=0.2,v0=0.08,k=0.7,rho=-0.4,theta=0.1,r=0.02,S0=250,q=0):
    Lambda = np.sqrt((sigma ** 2) * (complex(u ** 2, u)) + (complex(k, -rho * sigma * u)) ** 2)
    omega = np.exp(complex(0, u * (np.log(S0) + (r - q) * t)) + k * theta * t * complex(k, -rho * sigma * u) / sigma ** 2) \
            / (np.cosh(Lambda * t / 2) + (complex(k, -rho * sigma * u) / Lambda) * np.sinh(Lambda * t / 2)) ** (2 * k * theta / sigma ** 2)
    phi = omega * np.exp(-(complex(u ** 2, u)) * v0 / (Lambda / np.tanh(Lambda * t / 2) + complex(k, -rho * sigma * u)))
    return phi


def fft(K=250, n=10, alpha=1, UB=500, t=0.5, sigma=0.2, v0=0.08, k=.7, rho=-0.4, theta=0.1, r=0.015, q=0.0177,
        S0=267.15):
    N = 2 ** n
    B = UB

    v = np.linspace(0, B, N + 1)[0:-1]
    delta_v = B / N

    delta_k = 2 * np.pi / B
    beta = np.log(S0) - delta_k * N / 2
    km = beta + (np.linspace(0, N, N + 1)[0:-1]) * delta_k

    x = []
    for j in range(N):
        if j == 0:
            wj = 0.5
        else:
            wj = 1

        phi = characteristic_function(complex(v[j], -(alpha + 1)), t, sigma, v0, k, rho, theta, r, S0, q)
        xj = phi * np.exp(complex(0, -beta) * v[j]) * wj * delta_v / \
             (complex(alpha, v[j])) / complex(alpha + 1, v[j])
        x.append(xj)
    x = np.array(x) * np.exp(-r * t)
    y = np.fft.fft(x)

    call_price = []
    for j in range(N):
        c = np.exp(-alpha * (np.log(S0) - delta_k * (N / 2 - j))) * np.real(y[j]) / np.pi
        call_price.append(c)

    return np.exp(km), call_price

def call_price(sigma, v0, k, rho, theta,K=250,n=10,alpha=1,UB=600,t=0.5,r = 0.015,q = 0.0177,S0 = 267.15):
    K_list, price_list = fft(K,n,alpha,UB,t,sigma, v0, k, rho, theta,r,q,S0)
    return np.interp(K, K_list, price_list)

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

def call_rate_change(L):
    return all( x>=1 or x<=0 for x in (L/L.shift(1) - 1).fillna(0))

def put_rate_change(L):
    return all( x>=0 or x<=-1 for x in (L/L.shift(1) - 1).fillna(0))

def convex(L):
    return all( x>=0 for x in (((L.shift(-1) + L.shift(1))/2 - L).dropna()))

def optimizer(sigma, v0, k, rho, theta,dataset):
    T = np.unique(data.expT)
    sse = 0
    for t in T:
        K = data.K[data.expT == t]
        c = data.call_mid[data.expT==t]
        K_list,c_list = fft(sigma=sigma,v0=v0,k=k,rho=rho,theta=theta,t=t)
        sse += np.sum((np.interp(K, K_list, c_list) - c.tolist())**2)
    print(sse)
    return sse

# def optimizer(sigma, v0, k, rho, theta,dataset):
#     i = 0
#     sse = 0
#     while i< dataset.shape[0]:
#         T = dataset.expT[i]
#         strike = dataset.K[i]
#         c1 = call_price(sigma, v0, k, rho, theta,K=strike,n=10,alpha=1,UB=600,t=T)
#         c = dataset.call_mid[i]
#         sse += (c1 - c)**2
#         i = i + 1
#     print(sse)
#     return sse

def weighted_optimizer(sigma, v0, k, rho, theta,dataset):
    T = np.unique(data.expT)
    sse = 0
    for t in T:
        K = data.K[data.expT == t]
        c = data.call_mid[data.expT==t]
        w = 1/(dataset.call_ask[data.expT==t] - dataset.call_bid[data.expT==t])
        K_list,c_list = fft(sigma=sigma,v0=v0,k=k,rho=rho,theta=theta,t=t)
        sse += np.sum(w.tolist()*((np.interp(K, K_list, c_list)-c.tolist())**2))
    print(sse)
    return sse/10

# def weighted_optimizer(sigma, v0, k, rho, theta,dataset):
#     i = 0
#     sse = 0
#     while i< dataset.shape[0]:
#         T = dataset.expT[i]
#         strike = dataset.K[i]
#         w = 1/(dataset.call_ask[i] - dataset.call_bid[i])
#         c1 = call_price(sigma, v0, k, rho, theta,K=strike,n=10,alpha=1,UB=600,t=T)
#         c = dataset.call_mid[i]
#         sse += w*(c1 - c)**2
#         i = i + 1
#     print(sse)
#     return sse

def bsm_call_value(s0, k, t, r, sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    value = (s0 * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2))
    # print('cvalue',value)
    return value

def call_vega(K,sigma,T,S0=100,r=0):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) \
             / (sigma * np.sqrt(T))
    return S0*stats.norm.pdf(d1)*np.sqrt(T)

if __name__ == '__main__':
    #1.a
    put_par_list = [[0.1, 0.3225], [0.25, 0.2473], [0.4, 0.2021], [0.5, 0.1824]]
    call_par_list = [[0.4, 0.1574], [0.25, 0.1370], [0.1, 0.1148]]
    call_strike = []
    put_strike = []
    for par in call_par_list:
        call_strike.append(fsolve(lambda x: call_delta(x, par[1], 1 / 12, 100, 0) - par[0], x0=100)[0])
    for par in put_par_list:
        put_strike.append(fsolve(lambda x: put_delta(x, par[1], 1 / 12, 100, 0) - par[0], x0=100)[0])
    strike_1m = put_strike + call_strike
    sigma_1m = [i[1] for i in put_par_list + call_par_list]

    put_par_list = [[0.1, 0.2836], [0.25, 0.2178], [0.4, 0.1818], [0.5, 0.1645]]
    call_par_list = [[0.4, 0.1462], [0.25, 0.1256], [0.1, 0.1094]]

    call_strike = []
    put_strike = []
    for par in call_par_list:
        call_strike.append(fsolve(lambda x: call_delta(x, par[1], 3 / 12, 100, 0) - par[0], x0=100)[0])
    for par in put_par_list:
        put_strike.append(fsolve(lambda x: put_delta(x, par[1], 3 / 12, 100, 0) - par[0], x0=100)[0])
    strike_3m = put_strike + call_strike
    sigma_3m = [i[1] for i in put_par_list + call_par_list]

    K = {'1M': strike_1m, '3M': strike_3m}
    strike = pd.DataFrame(K, index=['10DP', '25DP', '40DP', '50D', '40DC', '25DC', '10DC'])

    #1.b
    df = {'sigma1M': sigma_1m, 'strike1M': strike_1m}
    df = pd.DataFrame(df)
    model = ols.OLS(y=df.sigma1M, x=df.strike1M)
    alpha1m = model.alpha
    beta1m = model.beta

    df = {'sigma3M': sigma_3m, 'strike3M': strike_3m}
    df = pd.DataFrame(df)
    model = ols.OLS(y=df.sigma3M, x=df.strike3M)
    alpha3m = model.alpha
    beta3m = model.beta

    #1.c
    K_list = np.linspace(80, 110, 301)
    vol_1M = alpha1m + beta1m * K_list
    vol_3M = alpha3m + beta3m * K_list
    c_1M = call_price(K_list, vol_1M, 1 / 12)
    c_3M = call_price(K_list, vol_3M, 3 / 12)

    j = 1
    p_1M = []
    p_3M = []
    while j < len(c_1M) - 1:
        p_1M.append((c_1M[j - 1] - 2 * c_1M[j] + c_1M[j + 1]) / (0.01))
        p_3M.append((c_3M[j - 1] - 2 * c_3M[j] + c_3M[j + 1]) / (0.01))
        j += 1

    plt.plot(K_list[1:-1], p_1M, label='1M')
    plt.plot(K_list[1:-1], p_3M, label='3M')
    plt.legend()
    plt.show()

    K_list = np.linspace(70, 130, 601)
    vol_1M = [0.1824 for i in range(len(K_list))]
    vol_3M = [0.1645 for i in range(len(K_list))]
    c_1M = call_price(K_list, vol_1M, 1 / 12)
    c_3M = call_price(K_list, vol_3M, 3 / 12)

    j = 1
    p_1M = []
    p_3M = []
    while j < len(c_1M) - 1:
        p_1M.append((c_1M[j - 1] - 2 * c_1M[j] + c_1M[j + 1]) / (0.01))
        p_3M.append((c_3M[j - 1] - 2 * c_3M[j] + c_3M[j + 1]) / (0.01))
        j += 1

    plt.plot(K_list[1:-1], p_1M, label='1M')
    plt.plot(K_list[1:-1], p_3M, label='3M')
    plt.legend()
    plt.show()

    #1.d
    K_list = np.linspace(80, 140, 601)
    vol_1M = alpha1m + beta1m * K_list
    c_1M = call_price(K_list, vol_1M, 1 / 12)

    j = 1
    d_1M = []
    while j < len(c_1M) - 1:
        d_1M.append((c_1M[j - 1] - 2 * c_1M[j] + c_1M[j + 1]) / (0.01))
        j += 1

    print(digital_put(110, K_list[1:-1], d_1M, 0.1))

    K_list = np.linspace(90, 120, 601)
    vol_3M = alpha3m + beta3m * K_list
    c_3M = call_price(K_list, vol_3M, 3 / 12)

    j = 1
    d_3M = []
    while j < len(c_3M) - 1:
        d_3M.append((c_3M[j - 1] - 2 * c_3M[j] + c_3M[j + 1]) / (0.01))
        j += 1

    print(digital_call(105, K_list[1:-1], d_3M, 0.2))

    K_list = np.linspace(80, 140, 601)
    vol_1M = alpha1m + beta1m * K_list
    c_1M = call_price(K_list, vol_1M, 1 / 12)
    j = 1
    d_1M = []
    while j < len(c_1M) - 1:
        d_1M.append((c_1M[j - 1] - 2 * c_1M[j] + c_1M[j + 1]) / (0.01))
        j += 1

    K_list = np.linspace(80, 140, 601)
    vol_3M = alpha3m + beta3m * K_list
    c_3M = call_price(K_list, vol_3M, 3 / 12)
    j = 1
    d_3M = []
    while j < len(c_3M) - 1:
        d_3M.append((c_3M[j - 1] - 2 * c_3M[j] + c_3M[j + 1]) / (0.01))
        j += 1

    d_2M = [0.5 * d_1M[i] + 0.5 * d_3M[i] for i in range(len(d_1M))]

    print(call(100,K_list[1:-1],d_2M,0.1))

    #2.a
    data = pd.read_excel('mf796-hw5-opt-data.xlsx')
    print(data.groupby(['expDays']).call_bid.apply(monotonic))
    print(data.groupby(['expDays']).call_ask.apply(monotonic))
    print(data.groupby(['expDays']).put_bid.apply(monotonic))
    print(data.groupby(['expDays']).put_ask.apply(monotonic))

    print(data.groupby(['expDays']).call_bid.apply(call_rate_change))
    print(data.groupby(['expDays']).call_ask.apply(call_rate_change))
    print(data.groupby(['expDays']).put_bid.apply(put_rate_change))
    print(data.groupby(['expDays']).put_ask.apply(put_rate_change))

    print(data.groupby(['expDays']).call_bid.apply(convex))
    print(data.groupby(['expDays']).call_ask.apply(convex))
    print(data.groupby(['expDays']).put_bid.apply(convex))
    print(data.groupby(['expDays']).put_ask.apply(convex))

    data['call_mid'] = (data['call_bid'] + data['call_ask']) / 2
    x0 = [2, 0.2, 0.5, -1, 0.1]
    bnds = ((0.01, 5), (0, 2), (0, 1), (-1, 1), (0, 1))
    args1 = minimize(lambda p: optimizer(p[0], p[1], p[2], p[3], p[4], data), x0, method='SLSQP', bounds=bnds)
    print(args1.fun)
    print(args1.x)
    x0 = [2, 0.2, 0.5, -1, 0.1]
    bnds = ((0.01, 2.5), (0, 1), (0, 1), (-1, 0.5), (0, 0.5))
    args2 = minimize(lambda p: optimizer(p[0], p[1], p[2], p[3], p[4], data), x0, method='SLSQP', bounds=bnds)
    print(args2.fun)
    print(args2.x)
    x0 = [0.5, 0.2, 0.2, 0, 0.2]
    bnds = ((0.01, 5), (0.01, 2), (0, 2), (-1, 1), (0, 1))
    args3 = minimize(lambda p: optimizer(p[0], p[1], p[2], p[3], p[4], data), x0, method='SLSQP', bounds=bnds)
    print(args3.fun)
    print(args3.x)
    x0 = [2, 0.2, 0.5, -1, 0.1]
    bnds = ((0.01, 2.5), (0, 1), (0, 1), (-1, 0.5), (0, 0.5))
    args11 = minimize(lambda p: weighted_optimizer(p[0], p[1], p[2], p[3], p[4], data), x0, method='SLSQP', bounds=bnds)
    print(args11.fun*10)
    print(args11.x)
    x0 = [2, 0.2, 0.5, -1, 0.1]
    bnds = ((0.01, 5), (0, 2), (0, 1), (-1, 1), (0, 1))
    args22 = minimize(lambda p: weighted_optimizer(p[0], p[1], p[2], p[3], p[4], data), x0, method='SLSQP', bounds=bnds)
    print(args22.fun*10)
    print(args22.x)
    x0 = [0.5, 0.2, 0.2, 0, 0.2]
    bnds = ((0.01, 5), (0.01, 2), (0, 2), (-1, 1), (0, 1))
    args33 = minimize(lambda p: weighted_optimizer(p[0], p[1], p[2], p[3], p[4], data), x0, method='SLSQP', bounds=bnds)
    print(args33.fun*10)
    print(args33.x)
    #3.a
    k = 3.51
    theta = 0.052
    sigma = 1.17
    rho = -0.77
    v0 = 0.034
    S0 = 267.15
    r = 0.015
    q = 0.0177
    T = 0.25
    K = 275
    alpha = 1
    n = 15
    UB = 1000
    c0 = call_price(sigma, v0, k, rho, theta, K, n, alpha, UB, T, r, q, S0)
    cplus = call_price(sigma, v0, k, rho, theta, K, n, alpha, UB, T, r, q, S0 + 2)
    cminus = call_price(sigma, v0, k, rho, theta, K, n, alpha, UB, T, r, q, S0 - 2)
    delta = (cplus - cminus) / 4
    print(delta)
    implied_vol = fsolve(lambda x: bsm_call_value(S0, K, T, r, x) - c0, x0=0.5)[0]
    bsm_delta = call_delta(K, implied_vol, T, S0, r)
    print(bsm_delta)
    #3.b
    cplus1 = call_price(sigma, v0 + 0.01, k, rho, theta + 0.01, K, n, alpha, UB, T, r, q, S0)
    cminus1 = call_price(sigma, v0 - 0.01, k, rho, theta - 0.01, K, n, alpha, UB, T, r, q, S0)
    Vega = (cplus1 - cminus1) / 0.02
    print(Vega)
    bsm_vega = call_vega(K, implied_vol, T, S0, r)
    print(bsm_vega)