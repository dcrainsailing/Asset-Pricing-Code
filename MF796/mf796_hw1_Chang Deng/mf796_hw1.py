import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class european_option(object):
    '''
    Default is European call option
    '''

    def __init__(self, S0=100, r=0.0, sigma=0.25, beta=1.0, T=1, steps=100, K=100, frequency=2000):
        self.S0 = S0
        self.S = np.zeros((steps + 1, frequency))
        self.S[0] = S0
        self.r = r
        self.sigma = sigma
        self.beta = beta
        self.T = T
        self.steps = steps
        self.K = K
        self.frequency = frequency
        self.dt = T / steps

    def simulation_paths(self):
        np.random.seed(2)
        for t in range(1, self.steps + 1):
            z = np.random.standard_normal(self.frequency)
            self.S[t] = self.S[t-1] + self.S[t-1]*self.r*self.dt \
                        + (self.S[t-1]**self.beta)*self.sigma*np.sqrt(self.dt)*z
            # self.S[t] = self.S[t - 1] * np.exp((self.r - 0.5 * self.sigma **2) * self.dt
            #                                    + self.sigma * z * np.sqrt(self.dt))
        return self

    def draw_paths(self):
        X = np.linspace(0, 1, self.steps + 1, endpoint=True)
        for i in range(self.frequency):
            plt.plot(X, self.S[:, i])
        plt.show()
        return

    def calc_simulated_price(self):
        return np.exp(-self.r*self.T) * np.mean(np.maximum(self.S[-1] - self.K,0))

    def calc_formula_price(self):
        self.d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = self.d1 - self.sigma * np.sqrt(self.T)
        return self.S0 * stats.norm.cdf(self.d1, 0, 1) - \
               self.K * np.exp(-self.r*self.T) * stats.norm.cdf(d2, 0, 1)

    def calc_delta(self):
        self.d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) \
                  / (self.sigma * np.sqrt(self.T))
        return stats.norm.cdf(self.d1, 0, 1)

    def simulate_portfolio(self,delta):
        payoff = np.maximum(self.S[-1] - self.K, 0) \
                 - delta * (self.S[-1] - self.S0)

        # plt.title('payoff after hedging')
        # plt.scatter(self.S[-1], payoff-self.calc_formula_price())
        # plt.show()
        # plt.title('payoff distribution after hedging')
        # plt.hist(payoff-self.calc_formula_price(), bins=100, density=True)
        # plt.show()
        #
        # plt.title('payoff without hedging')
        # plt.scatter(self.S[-1], np.maximum(0, self.S[-1] - self.K))
        # plt.show()
        # plt.title('payoff distribution without hedging')
        # plt.hist(np.maximum(0, self.S[-1] - self.K), bins=100, density=True)
        # plt.show()
        return np.mean(payoff)


if __name__ == "__main__":
    e = european_option()
    e.simulation_paths()
    e.draw_paths()
    print('simulated_price:',e.calc_simulated_price())
    print('formula_price:',e.calc_formula_price())
    print('delta:',e.calc_delta())
    print('portfolio_payoff:',e.simulate_portfolio(e.calc_delta()))

    e1 = european_option(beta=0.5)
    e1.simulation_paths()
    print('portfolio_payoff:',e1.simulate_portfolio(e1.calc_delta()))

    e2 = european_option(sigma=0.4)
    e2.simulation_paths()
    print('portfolio_payoff:',e2.simulate_portfolio(e2.calc_delta()))
    print('formula_price:', e2.calc_formula_price())

    pf_list = []
    sg_list = []
    for sg in np.arange(0.01,1,0.01):
        sg_list += [sg]
        op = european_option(sigma=sg)
        pf_list += [op.simulate_portfolio(op.calc_delta())]
    plt.plot(sg_list, pf_list)
    plt.show()