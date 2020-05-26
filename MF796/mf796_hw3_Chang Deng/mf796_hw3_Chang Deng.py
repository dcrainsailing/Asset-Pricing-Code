import numpy as np
from scipy.fftpack import fft,ifft
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def characteristic_function(u,t,sigma=0.2,v0=0.08,k=0.7,rho=-0.4,theta=0.1,r=0.02,S0=250,q=0):
    Lambda = np.sqrt((sigma ** 2) * (complex(u ** 2, u)) + (complex(k, -rho * sigma * u)) ** 2)
    omega = np.exp(complex(0, u * (np.log(S0) + (r - q) * t)) + k * theta * t * complex(k, -rho * sigma * u) / sigma ** 2) \
            / (np.cosh(Lambda * t / 2) + (complex(k, -rho * sigma * u) / Lambda) * np.sinh(Lambda * t / 2)) ** (2 * k * theta / sigma ** 2)
    phi = omega * np.exp(-(complex(u ** 2, u)) * v0 / (Lambda / np.tanh(Lambda * t / 2) + complex(k, -rho * sigma * u)))
    return phi


def fft(K=250, n=10, alpha=1, UB=500, t=0.5):
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

def call_price(K=250,n=10,alpha=1,UB=600,t=0.5):
    K_list, price_list = fft(K,n,alpha,UB,t)
    return np.interp(K, K_list, price_list)

def bsm_call_value(s0, k, t, r, sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    value = (s0 * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2))
    # print('cvalue',value)
    return value

def impliedvol_to_K(n=10,alpha=1,UB=600,t=0.25):
    K_list = np.linspace(80,230,60)
    c0_list = []
    strike_list, price_list = fft(K=150,n=n,alpha=alpha,UB=UB,t=t)
    for k in K_list:
        c0 = np.interp(k, strike_list, price_list)
        c0_list.append(c0)
    #plt.plot(K_list, c0_list)
    vol_list = []
    for i in np.arange(0,len(K_list),1):
        vol_list.append(fsolve(lambda x: bsm_call_value(150, K_list[i], 0.25, 0.025, x)-c0_list[i],x0=0.5))
    return np.array(K_list),np.array(vol_list)

def impliedvol_to_T(K=150,n=10,alpha=1,UB=600,t=0.25):
    t_list = np.arange(0.1,2,0.1)
    c0_list = []
    for ti in t_list:
        c0_list.append(call_price(K=K,n=n,alpha=alpha,UB=UB,t=ti))
    #plt.plot(t_list, c0_list)
    vol_list = []
    for i in np.arange(0,len(t_list),1):
        vol_list.append(fsolve(lambda x: bsm_call_value(150, 150, t_list[i], 0.025, x)-c0_list[i],x0=0.5))
    return np.array(t_list),np.array(vol_list)

def change_sigma():
    global sigma
    sigma = 0.4
    sigma_list = np.arange(-0.3, 0.2, 0.1) + sigma
    strike_list = []
    vol_list = []
    i = 0
    for si in sigma_list:
        sigma = si
        strike, Vol = impliedvol_to_K()
        strike_list.append(strike)
        vol_list.append(Vol)
    plt.title('Volatility skews')
    plt.xlabel('Strike price')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(strike_list[j], vol_list[j], label='sigma=' + str(round(sigma_list[j], 2)) + '')
    plt.legend()
    plt.show()

    sigma = 0.4
    sigma_list = np.arange(-0.3, 0.2, 0.1) + sigma
    t_list = []
    vol_list = []
    i = 0
    for si in sigma_list:
        sigma = si
        ti, Vol = impliedvol_to_T()
        t_list.append(ti)
        vol_list.append(Vol)
    plt.title('Term Structure')
    plt.xlabel('time to expiry')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(t_list[j], vol_list[j], label='sigma=' + str(round(sigma_list[j], 2)) + '')
    plt.legend()
    plt.show()
    return

def change_v0():
    global sigma,v0
    sigma = 0.4
    v0 = 0.09
    v0_list = np.arange(-0.02, 0.03, 0.01) + v0
    strike_list = []
    vol_list = []
    i = 0
    for v0i in v0_list:
        v0 = v0i
        strike, Vol = impliedvol_to_K()
        strike_list.append(strike)
        vol_list.append(Vol)
    plt.title('Volatility skews')
    plt.xlabel('Strike price')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(strike_list[j], vol_list[j], label='v0=' + str(round(v0_list[j], 2)) + '')
    plt.legend()
    plt.show()

    v0 = 0.09
    v0_list = np.arange(-0.02, 0.03, 0.01) + v0
    t_list = []
    vol_list = []
    i = 0
    for v0i in v0_list:
        v0 = v0i
        ti, Vol = impliedvol_to_T()
        t_list.append(ti)
        vol_list.append(Vol)
    plt.title('Term Structure')
    plt.xlabel('time to expiry')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(t_list[j], vol_list[j], label='v0=' + str(round(v0_list[j], 2)) + '')
    plt.legend()
    plt.show()

    return

def change_k():
    global sigma,v0,k
    sigma = 0.4
    v0 = 0.09
    k = 0.5
    k_list = np.arange(-0.2, 0.3, 0.1) + k
    strike_list = []
    vol_list = []
    i = 0
    for ki in k_list:
        k = ki
        strike, Vol = impliedvol_to_K()
        strike_list.append(strike)
        vol_list.append(Vol)
    plt.title('Volatility skews')
    plt.xlabel('Strike price')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(strike_list[j], vol_list[j], label='k=' + str(round(k_list[j], 2)) + '')
    plt.legend()
    plt.show()

    v0 = 0.09
    k = 0.5
    k_list = np.arange(-0.2, 0.3, 0.1) + k
    t_list = []
    vol_list = []
    i = 0
    for ki in k_list:
        k = ki
        ti, Vol = impliedvol_to_T()
        t_list.append(ti)
        vol_list.append(Vol)
    plt.title('Term Structure')
    plt.xlabel('time to expiry')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(t_list[j], vol_list[j], label='v0=' + str(round(k_list[j], 2)) + '')
    plt.legend()
    plt.show()

    return

def change_rho():
    global sigma,v0,k,rho
    sigma = 0.4
    v0 = 0.09
    k = 0.5
    rho = 0.25
    rho_list = np.arange(-0.75, 0.75, 0.25) + rho
    strike_list = []
    vol_list = []
    i = 0
    for rhoi in rho_list:
        rho = rhoi
        strike, Vol = impliedvol_to_K()
        strike_list.append(strike)
        vol_list.append(Vol)
    plt.title('Volatility skews')
    plt.xlabel('Strike price')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(strike_list[j], vol_list[j], label='rho=' + str(round(rho_list[j], 2)) + '')
    plt.legend()
    plt.show()

    v0 = 0.09
    k = 0.5
    rho = 0.25
    rho_list = np.arange(-0.75, 0.75, 0.25) + rho
    t_list = []
    vol_list = []
    i = 0
    for rhoi in rho_list:
        rho = rhoi
        ti, Vol = impliedvol_to_T()
        t_list.append(ti)
        vol_list.append(Vol)
    plt.title('Term Structure')
    plt.xlabel('time to expiry')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(t_list[j], vol_list[j], label='rho=' + str(round(rho_list[j], 2)) + '')
    plt.legend()
    plt.show()

    return

def change_theta():
    global sigma,v0,k,rho,theta
    sigma = 0.4
    v0 = 0.09
    k = 0.5
    rho = 0.25
    theta = 0.12
    theta_list = np.arange(-0.1, 0.2, 0.1) + theta
    strike_list = []
    vol_list = []
    i = 0
    for thetai in theta_list:
        theta = thetai
        strike, Vol = impliedvol_to_K()
        strike_list.append(strike)
        vol_list.append(Vol)
    plt.title('Volatility skews')
    plt.xlabel('Strike price')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(strike_list[j], vol_list[j], label='theta=' + str(round(theta_list[j], 2)) + '')
    plt.legend()
    plt.show()

    v0 = 0.09
    k = 0.5
    rho = 0.25
    theta = 0.12
    theta_list = np.arange(-0.01, 0.02, 0.01) + theta
    t_list = []
    vol_list = []
    i = 0
    for thetai in theta_list:
        theta = thetai
        ti, Vol = impliedvol_to_T()
        t_list.append(ti)
        vol_list.append(Vol)
    plt.title('Term Structure')
    plt.xlabel('time to expiry')
    plt.ylabel('Volatility')
    l = len(vol_list)
    for j in np.arange(0, l, 1):
        plt.plot(t_list[j], vol_list[j], label='theta=' + str(round(theta_list[j], 2)) + '')
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    # question a
    sigma = 0.2
    v0 = 0.08
    k = 0.7
    rho = -0.4
    theta = 0.1
    r = 0.02
    S0 = 250
    q = 0

    # question a.i
    cp = call_price()
    alpha_list = [0.01, 0.02, 0.25, 0.5, 0.8, 1, 1.05, 1.5, 1.75, 10, 30]
    c0_list = []
    for i in alpha_list:
        c0_list.append(call_price(alpha=i))
    plt.xlabel('alpha')
    plt.ylabel('Price')
    plt.plot(alpha_list, c0_list)
    plt.show()
    print(dict(zip(alpha_list, c0_list)))

    alpha_list = [0.01, 0.02, 0.25, 0.5, 0.8, 1, 1.05, 1.5, 1.75, 2]
    c0_list = []
    for i in alpha_list:
        c0_list.append(call_price(alpha=i))
    plt.xlabel('alpha')
    plt.ylabel('Price')
    plt.plot(alpha_list, c0_list)
    plt.show()

    # question a.ii
    n_list = np.arange(5, 13, 1)
    c0_list = []
    for i in n_list:
        c0_list.append(call_price(alpha=1, n=i))
    plt.title('Error')
    plt.xlabel('2^N')
    plt.ylabel('Price')
    plt.plot(n_list, c0_list)
    plt.show()

    UB_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    c0_list = []
    for i in UB_list:
        c0_list.append(call_price(UB=i))
    plt.title('Error')
    plt.xlabel('B')
    plt.ylabel('Price')
    plt.plot(UB_list, c0_list)
    plt.show()

    n_list = []
    UB_list = []
    error_list = []
    runtime_list = []
    for i in np.arange(10, 200, 10):
        for j in np.arange(9, 12, 1):
            start = time.time()
            c = call_price(alpha=1, UB=i, n=j)
            over = time.time()
            if abs(c - cp) < 0.01:
                n_list.append(j)
                UB_list.append(i)
                error_list.append(abs(c - cp))
                runtime_list.append(np.float64(over - start))
    ax = plt.subplot(111, projection='3d')
    plt.title('Error')
    plt.xlabel('B')
    plt.ylabel('N')
    ax.plot_trisurf(UB_list, n_list, error_list)
    plt.show()

    ax = plt.subplot(111, projection='3d')
    plt.title('Efficiency')
    plt.xlabel('B')
    plt.ylabel('N')
    ax.plot_trisurf(UB_list, n_list, runtime_list)
    plt.show()

    # question a.iii
    cp = call_price(K=260)
    n_list = []
    c0_list = []
    for i in np.arange(5, 13, 1):
        n_list.append(i)
        c0_list.append(call_price(K=260, n=i))
    plt.title('Error')
    plt.xlabel('2^N')
    plt.ylabel('Price')
    plt.plot(n_list, c0_list)
    plt.show()

    UB_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    c0_list = []
    for i in UB_list:
        c0_list.append(call_price(UB=i))
    plt.title('Error')
    plt.xlabel('B')
    plt.ylabel('Price')
    plt.plot(UB_list, c0_list)
    plt.show()

    n_list = []
    UB_list = []
    error_list = []
    runtime_list = []
    for i in np.arange(10, 200, 10):
        for j in np.arange(9, 12, 1):
            start = time.time()
            c = call_price(K=260, alpha=1, UB=i, n=j)
            over = time.time()
            if abs(c - cp) < 0.05:
                n_list.append(j)
                UB_list.append(i)
                error_list.append(abs(c - cp))
                runtime_list.append(np.float64(over - start))
    ax = plt.subplot(111, projection='3d')
    plt.title('Error')
    plt.xlabel('B')
    plt.ylabel('N')
    ax.plot_trisurf(UB_list, n_list, error_list)
    plt.show()

    ax = plt.subplot(111, projection='3d')
    plt.title('Efficiency')
    plt.xlabel('B')
    plt.ylabel('N')
    ax.plot_trisurf(UB_list, n_list, runtime_list)
    plt.show()

    # question b
    sigma = 0.4
    v0 = 0.09
    k = 0.5
    rho = 0.25
    theta = 0.12
    r = 0.025
    S0 = 150
    q = 0

    # question b.i
    Strike, Vol_K = impliedvol_to_K()
    plt.title('Volatility Skews')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.plot(Strike, Vol_K)
    plt.show()

    # question b.ii
    tlist, Vol_T = impliedvol_to_T()
    plt.title('Term Structure')
    plt.xlabel('time to expiry')
    plt.ylabel('Volatility')
    plt.plot(tlist, Vol_T)
    plt.show()

    # question b.iii
    change_sigma()
    change_v0()
    change_k()
    change_rho()
    change_theta()

