import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulation(sigma=2.18, v0=0.05, k=4.11, rho=-0.8, theta=0.07, q=0.0177, r=0.015, S0=282, T=1, dt=1/252, paths=10000):
    steps = int(round(T / dt))
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
        v[t] = np.maximum(v[t-1] + k*(theta-v[t-1])*dt + sigma*np.sqrt(v[t-1])*x2,0) # Truncate
        #v[t] = np.abs(v[t-1] + k*(theta-v[t-1])*dt + sigma*np.sqrt(v[t-1])*x2) # Reflect
        S[t] = S[t-1] + (r-q)*S[t-1]*dt + np.sqrt(v[t-1])*S[t-1]*x1
#     X = np.linspace(0, 1, steps + 1, endpoint=True)
#     for i in range(N):
#         plt.plot(X, S[:, i])
#     plt.show()
    M = [max(S[:,i]) for i in range(N)]
    return S,M

def calc_simulated_price(r,T,S,K):
    payoff = np.maximum(S[-1] - K,0)
    return np.exp(-(r+q)*T) * np.mean(payoff)

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

def up_and_out(r,T,S,M,K1,K2):
    payoff = np.maximum(S[-1] - K1,0)
    sign = [m<K2 for m in M]
    return np.exp(-(r+q)*T)*np.mean(payoff*sign)

if __name__ == '__main__':
    # a
    sigma = 0.7
    v0 = 0.05
    k = 3.65
    rho = -0.8
    theta = 0.07
    q = 0.0177
    r = 0.015
    T = 1
    S0 = 282
    K = 285
    # b
    N = np.arange(100, 21000, 5000)
    dt = 1 / np.arange(500, 100, -100)
    N_list = []
    dt_list = []
    c_list = []
    j = 0
    for n in N:
        for d in dt:
            S, Mean = simulation(dt=d, paths=n)
            c0 = calc_simulated_price(r, T, S, K)
            N_list += [n]
            dt_list += [d]
            c_list += [c0]
    ax = plt.subplot(111, projection='3d')
    plt.title('Convergence')
    plt.xlabel('N')
    plt.ylabel('dt')
    ax.plot_trisurf(N_list, dt_list, c_list)
    plt.show()
    # c
    S, M = simulation(sigma=sigma, v0=v0, k=k, rho=rho, q=q, r=r, S0=S0)
    c0 = calc_simulated_price(r,T,S,K)
    print(c0)
    c1 = call_price(sigma=sigma, v0=v0, k=k, rho=rho, theta=theta, K=K, n=10, alpha=1, UB=600, t=T, r=r, q=q, S0=S0)
    print(c1)
    # d
    K1 = 285
    K2 = 315
    c = up_and_out(r, T, S, M, K1, K2)
    print(c)
    c1 = []
    N = np.arange(100, 21000, 1000)
    for n in N:
        S, M = simulation(dt=d, paths=n)
        c1 += [up_and_out(r, T, S, M, K1, K2)]
    plt.plot(N, c1)
    plt.show()
    # e
    K1 = 285
    K2 = 315
    Theta = []
    Theta_hat = []
    N = [10 ** i for i in range(3, 6)]
    times = 10
    j = 0
    i = 0
    theta = []
    z = []
    while j < len(N):
        n = N[j]
        i = 0
        while i < times:
            S, M = simulation(sigma=sigma, v0=v0, k=k, rho=rho, q=q, r=r, S0=S0, T=T, paths=n)
            hx = np.maximum(S[-1] - K1, 0) * [m < K2 for m in M]
            theta += [np.mean(hx)]
            euro_call = np.mean(np.maximum(S[-1] - K1, 0))
            z += [euro_call]
            i += 1
        c = -np.cov(theta, z)[0][1] / np.var(z)
        print(np.sqrt((c ** 2) * np.var(z) / np.var(theta)))
        theta_hat = theta + c * (z - np.mean(z))
        Theta.append(np.var(theta))
        Theta_hat.append(np.var(theta_hat))
        j += 1
        theta.clear()
        z.clear()
    plt.plot(np.log(N), Theta, label='without control variate')
    plt.plot(np.log(N), Theta_hat, label='with control variate')
    plt.legend()
    plt.show()

    K1 = 285
    K2 = 500
    Theta = []
    Theta_hat = []
    N = [10 ** i for i in range(3, 6)]
    times = 10
    j = 0
    i = 0
    theta = []
    z = []
    while j < len(N):
        n = N[j]
        i = 0
        while i < times:
            S, M = simulation(sigma=sigma, v0=v0, k=k, rho=rho, q=q, r=r, S0=S0, T=T, paths=n)
            hx = np.maximum(S[-1] - K1, 0) * [m < K2 for m in M]
            theta += [np.mean(hx)]
            euro_call = np.mean(np.maximum(S[-1] - K1, 0))
            z += [euro_call]
            i += 1
        c = -np.cov(theta, z)[0][1] / np.var(z)
        print(np.sqrt((c ** 2) * np.var(z) / np.var(theta)))
        theta_hat = theta + c * (z - np.mean(z))
        Theta.append(np.var(theta))
        Theta_hat.append(np.var(theta_hat))
        j += 1
        theta.clear()
        z.clear()
    plt.plot(np.log(N), Theta,label = 'without control variate')
    plt.plot(np.log(N), Theta_hat, label = 'with control variate')
    plt.legend()
    plt.show()
    