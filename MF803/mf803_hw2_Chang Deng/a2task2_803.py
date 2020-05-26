from a1task2_803 import *
import statsmodels.api as sm

class lookback_oprion_with_Bachelier_model(lookback_european_option):
    def __init__(self,S0=100,r=0,sigma=10,T=1,steps=100,K=100,frequency=2000):
        super(lookback_oprion_with_Bachelier_model,self).__init__(S0,r,sigma,T,steps,K,frequency)

    def simulation_paths(self):
        self.random_distribution()
        w = np.zeros((self.steps + 1, self.frequency))
        w[0] = 0

        for t in range(1, self.steps + 1):
            z = np.random.standard_normal(self.frequency)
            w[t] = w[t-1] + z*((self.dt)**(1/2))
            self.S[t] = self.S[t - 1] + self.r*self.dt + self.sigma*(w[t]-w[t-1])
        return self

    def draw_terminal_price_histogram(self):
        self.simulation_paths()
        St = self.S[self.steps]
        plt.hist(St, density=0, facecolor='blue', alpha=0.75)
        plt.show()
        return

    def examine_distribution(self):
        self.simulation_paths()
        St = self.S[self.steps]
        sm.qqplot(St,fit=True,line='45')
        plt.show()
        return

def varying_delta(begin,end,steps):
    epsilon_list = []
    delta_list = []

    for epsilon in np.arange(begin,end,steps):
        option1 = lookback_oprion_with_Bachelier_model(S0=100+epsilon)
        option2 = lookback_oprion_with_Bachelier_model(S0=100-epsilon)
        c1 = option1.simulated_price('put')
        c2 = option2.simulated_price('put')
        delta = (c1 - c2)/(2*epsilon)
        epsilon_list.append(epsilon)
        delta_list.append(delta)

    plt.plot(epsilon_list,delta_list)
    plt.show()
    list = {'Epsilon':epsilon_list,'Delta':delta_list}
    return pd.DataFrame(list)

if __name__ == '__main__':
    f = lookback_oprion_with_Bachelier_model(steps=100,sigma=10,frequency=10000)
    lb = lookback_european_option(steps=100,frequency=10000)
    f.draw_paths()
    #lb.draw_paths()
    f.draw_terminal_price_histogram()
    f.examine_distribution()
    bc_price = f.simulated_price()
    bs_price = lb.simulated_price()
    print(bs_price, bc_price)
    #before run varying_delta function,
    #please comment out the sentence "np.random.seed(1)"  in the random_distribution function of a1task2_803.py file
    print(varying_delta(0,1, 0.01))
    print(varying_delta(1,100,0.1))