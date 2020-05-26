import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr


def Euler_discretization(S0, K1, K2, Smin, Smax, M, T, N, r, sigma, right=False):
    ht = T / N
    hs = Smax / M
    Si = np.arange(Smin, Smax + hs, hs)  # j = 0,1,...,M-1

    ai = 1 - (sigma ** 2) * (Si ** 2) * (ht / (hs ** 2)) - r * ht
    li = ((sigma ** 2) * (Si ** 2) / 2) * (ht / (hs ** 2)) - r * Si * ht / 2 / hs
    ui = ((sigma ** 2) * (Si ** 2) / 2) * (ht / (hs ** 2)) + r * Si * ht / 2 / hs

    ai = ai[1:M]  # ai, i = 1,...,M-1
    li = li[2:M]  # li, i = 2,...,M-1
    ui = ui[1:M - 1]  # ui, i = 1,...,M-2

    A = np.diag(ai)
    x, y = A.nonzero()
    j_xy = ((x + 1)[0:-1], y[0:-1])
    u_xy = (x[0:-1], (y + 1)[0:-1])
    A[j_xy] = li
    A[u_xy] = ui

    eig_vals, eig_vecs = np.linalg.eig(A)
    for eig in eig_vals:
        if abs(eig) > 1:
            print('absolute eignvalue larger than 1')
            exit()
    plt.plot(-np.sort(-abs(eig_vals)))

    CT1 = Si - K1
    CT2 = K2 - Si
    CT1 = np.maximum(CT1, 0)
    CT2 = np.minimum(CT2, 0)
    CT = (CT1 + CT2)

    Ct = CT[1:M]
    for i in range(N):
        Ct = A.dot(Ct)
        Ct[-1] = Ct[-1] + ui[-1] * (K2 - K1) * np.exp(-r * i * ht)
        if right == True:
            Ct = [max(x, y) for x, y in zip(Ct, CT[1:M])]
    # print(Ct)
    # print(Si[1:M])
    c0 = np.interp(S0, Si[1:M], Ct)
    return c0

if __name__ == '__main__':
    r = 0.0072
    strike = [287, 291, 317, 353]
    implied_vol = [0.331, 0.3217, 0.2599, 0.2374]
    sigma = np.interp(317.5, strike, implied_vol)

    Smin = 0
    Smax = 200
    T = 0.5
    N = 10
    M = 10
    ht = T / N
    hs = Smax / M
    Si = np.arange(Smin, Smax, hs)  # j = 0,1,...,M-1
    ai = 1 - (sigma ** 2) * (Si ** 2) * ht / hs - r * ht
    li = ((sigma ** 2) * (Si ** 2) / 2) * (ht / (hs ** 2)) - r * Si * ht / 2 / hs
    ui = ((sigma ** 2) * (Si ** 2) / 2) * (ht / (hs ** 2)) + r * Si * ht / 2 / hs

    ai = ai[1:M]  # ai, i = 1,...,M-1
    li = li[2:M]  # li, i = 2,...,M-1
    ui = ui[1:M - 1]  # ui, i = 1,...,M-2

    A = np.diag(ai)
    x, y = A.nonzero()
    j_xy = ((x + 1)[0:-1], y[0:-1])
    u_xy = (x[0:-1], (y + 1)[0:-1])

    A[j_xy] = li
    A[u_xy] = ui

    S0 = pdr.get_data_yahoo('SPY', start='2020-03-04', end='2020-03-05')['Close'][
        0]  # SPY's closing price on March 4,2020.
    print(S0)
    K1 = 315
    K2 = 320
    Smin = 0
    Smax = 500
    M = 250
    T = 7 / 12
    N = 4000
    spread = Euler_discretization(S0, K1, K2, Smin, Smax, M, T, N, r, sigma, right=False)
    spread_right = Euler_discretization(S0, K1, K2, Smin, Smax, M, T, N, r, sigma, right=True)
    print(spread)
    print(spread_right)
    print(spread_right - spread)