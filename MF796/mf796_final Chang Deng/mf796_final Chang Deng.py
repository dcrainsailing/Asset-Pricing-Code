import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import matrix, solvers
import scipy
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds

def simulation(sigma=0.25, v0=0.05**2, k=1, rho=-0.5, theta=0.1, q=0, r=0, S0=100, T=3/12, dt=1/252, paths=10000):
    steps = round(T/dt)
    S = np.zeros((steps + 1, paths))
    S[0] = S0
    v = np.zeros((steps + 1, paths))
    v[0] = v0
    N = paths
    for t in range(1, steps + 1):
        z1 = np.random.standard_normal(N)
        z2 = np.random.standard_normal(N)
        x1 = 0 + np.sqrt(dt)*z1
        x2 = 0 + np.sqrt(dt)*(rho*z1 + np.sqrt(1-rho**2)*z2)
        #v[t] = np.abs(v[t-1] + k*(theta-v[t-1])*dt + sigma*np.sqrt(v[t-1])*x2) # Reflect
        S[t] = S[t-1] + (r-q)*S[t-1]*dt + np.sqrt(v[t-1])*S[t-1]*x1
        v[t] = np.maximum(v[t-1] + k*(theta-v[t-1])*dt + sigma*np.sqrt(v[t-1])*x2,0) # Truncate
#     X = np.linspace(0, 1, steps + 1, endpoint=True)
#     for i in range(N):
#         plt.plot(X, S[:, i])
#     plt.show()
    Mean = [np.mean(S[:,i]) for i in range(N)]
    return S,Mean

def calc_simulated_price(r,T,Mean,K):
    payoff = np.maximum(np.array(Mean) - K,0)
    return np.exp(-(r+q)*T) * np.mean(payoff)

def mini_cov(w):
    R = pd.read_csv('DataForProblem4.csv')
    Ra = (R.iloc[:,1:11])
    C = (Ra.cov()).values
    return 10000*np.dot(w.T,np.dot(w,C))

def optimal_port(w):
    a = 0.5
    R = pd.read_csv('DataForProblem4.csv')
    Ra = (R.iloc[:,1:11])
    R_mean = list(Ra.mean())
    C = (Ra.cov()).values
    sum = np.dot(R_mean,w) - a*np.dot(np.dot(w,C),w)
    return -sum*100000

def max_return(w):
    R = pd.read_csv('DataForProblem4.csv')
    Ra = (R.iloc[:,1:11])
    R_mean = list(Ra.mean())
    ER = np.dot(R_mean,w)
    return -ER*100000

def cons_f(w):
    A = np.ones(10)
    return np.dot(A,w)

def bench_track(w):
    R = pd.read_csv('DataForProblem4.csv')
    Ra = (R.iloc[:,1:11])
    bench = R.loc[:,'B1']
    e = np.sum((np.dot(Ra,w) - bench)**2)
    return e*10000

def log_utility(w):
    R = pd.read_csv('DataForProblem4.csv')
    Ra = (R.iloc[:,1:11])
    col_name = Ra.columns.tolist()
    Ra.insert(col_name.index('Sec1'),'Cash',0)
    Ra = Ra + 1
    return 10000*(-np.sum(np.log(np.dot(Ra,w)))/Ra.shape[0])

if __name__ == '__main__':
    # 3.b
    sigma = 0.25
    v0 = 0.05 ** 2
    k = 1
    rho = -0.5
    theta = 0.1
    q = 0
    r = 0
    T = 3 / 12
    S0 = 100
    K = 100
    S, Mean = simulation(sigma=sigma, v0=v0, k=k, rho=rho, q=q, r=r, S0=S0)
    c0 = calc_simulated_price(r, T, Mean, K)
    print(c0)

    # 3.c
    N = np.arange(10, 100000, 10000)
    c = []
    On12 = []
    j = 0
    for n in N:
        S, Mean = simulation(paths=n)
        c0 = calc_simulated_price(r, T, Mean, K)
        n12 = n ** (-1 / 2)
        c.append(c0)
        On12.append(n12)

    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.plot(N, c, color='red')
    ax2 = ax1.twinx()
    ax2.plot(N, On12, label='O(n^(-0.5))', color='blue')
    ax2.legend()
    plt.show()

    # 3.ef
    Theta = []
    Theta_hat = []
    N = np.arange(10, 100000, 10000)
    times = 100
    theta = []
    z = []
    for n in N:
        for i in range(times):
            S, Mean = simulation(sigma=sigma, v0=v0, k=k, rho=rho, q=q, r=r, S0=S0, T=T, paths=n)
            hx = np.maximum(np.array(Mean) - K, 0)
            theta += [np.mean(hx)]
            euro_call = np.mean(np.maximum(S[-1] - K, 0))
            z += [euro_call]
        c = -np.cov(theta, z)[0][1] / np.var(z)
        print(c)
        print(np.sqrt((c**2)*np.var(z)/np.var(theta)))
        theta_hat = theta + c * (z - np.mean(z))
        Theta.append(np.var(theta))
        Theta_hat.append(np.var(theta_hat))
        theta.clear()
        z.clear()
    plt.plot(np.log(N), np.log(Theta), label='without_control variate')
    plt.plot(np.log(N), np.log(Theta_hat), label='with control variate')
    plt.legend()
    plt.show()

    # 4.a
    x0 = [0.1] * 10
    A = np.ones(10)
    L = [0] * 10
    U = [1] * 10
    bounds = Bounds(L, U)
    linear_constraint = LinearConstraint(A, [1], [1])
    w1 = scipy.optimize.minimize(mini_cov, x0, constraints=[linear_constraint], bounds=bounds)
    print(w1)

    # 4.b
    x0 = [0.1] * 10
    A = np.ones(10)
    L = [0] * 10
    U = [1] * 10
    bounds = Bounds(L, U)
    linear_constraint = LinearConstraint(A, [1], [1])
    w2 = scipy.optimize.minimize(optimal_port, x0, constraints=[linear_constraint], bounds=bounds)
    print(w2)

    # 4.c
    x0 = [0.1] * 10
    L = [0] * 10
    U = [1] * 10
    bounds = Bounds(L, U)
    # linear_constraint = LinearConstraint(A,[1],[1])
    nonlinear_constraint = NonlinearConstraint(cons_f, 0, 1)
    w3 = scipy.optimize.minimize(max_return, x0, constraints=[nonlinear_constraint], bounds=bounds)
    print(w3)

    # 4.d
    x0 = [0.1] * 10
    A = np.ones(10)
    L = [0] * 10
    U = [1] * 10
    bounds = Bounds(L, U)
    linear_constraint = LinearConstraint(A, [1], [1])
    w4 = scipy.optimize.minimize(bench_track, x0, constraints=[linear_constraint], bounds=bounds)
    print(w4)

    # 4.f
    x0 = [1 / 11] * 11
    A = np.ones(11)
    L = [0] * 11
    U = [1] * 11
    bounds = Bounds(L, U)
    linear_constraint = LinearConstraint(A, [1], [1])
    w5 = scipy.optimize.minimize(log_utility, x0, constraints=[linear_constraint], bounds=bounds)
    print(w5)