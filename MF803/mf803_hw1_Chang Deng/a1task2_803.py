# -*- coding: utf-8 -*-
"""
File name: a1task1.py
Name: Chang Deng
Email address: dengc@bu.edu
Assignment number: 1
Description: python code of task 2, assignment 1
Exotic Option Pricing via Simulation
"""
from math import exp, sqrt, log
from random import gauss, seed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class european_option(object):

    def __init__(self,S0=100,r=0,sigma=0.25,T=1,steps=100,K=100,frequency=2000):
        self.S0 = S0
        self.S = np.zeros((steps + 1, frequency))
        self.S[0] = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.K = K
        self.frequency = frequency
        self.dt = T/steps

    def random_distribution(self):
        np.random.seed(1)
        self.z = np.random.standard_normal(self.frequency)
        return self

    def simulation_paths(self):
        self.random_distribution()
        z = self.z
        for t in range(1, self.steps + 1):
            z = np.random.standard_normal(self.frequency)
            self.S[t] = self.S[t - 1] * np.exp((self.r-0.5*self.sigma**2)*self.dt \
                                               + self.sigma * sqrt(self.dt) * z)
        return self

    def draw_paths(self):
        X = np.linspace(0, 1, self.steps + 1, endpoint=True)
        self.simulation_paths()
        for i in range(self.frequency):
            plt.plot(X, self.S[:, i])
        plt.show()
        return

    def terminal_mean(self):
        self.simulation_paths()
        mean = np.mean(self.S[self.steps])
        return mean

    def terminal_var(self):
        self.simulation_paths()
        var = np.var(self.S[self.steps])
        return var

    def call_payoffs(self):
        self.simulation_paths()
        self.c0 = np.maximum(self.S[self.steps] - self.K, 0)
        return self

    def put_payoffs(self):
        self.simulation_paths()
        self.p0 = np.maximum(self.K - self.S[self.steps], 0)
        return self

    def payoffs_mean(self,position:str = 'put'):
        if position == 'call':
            self.call_payoffs()
            mean = np.mean(self.c0)
        if position == 'put':
            self.put_payoffs()
            mean = np.mean(self.p0)
        return mean

    def payoffs_std(self,position:str = 'put'):
        if position == 'call':
            self.call_payoffs()
            var = np.std(self.c0)
        if position == 'put':
            self.put_payoffs()
            var = np.std(self.p0)
        return var

    def draw_payoffs_histogram(self,position:str = 'put'):
        if position == 'call':
            self.call_payoffs()
            plt.hist(self.c0, density=0, facecolor='blue', alpha=0.75)
        if position == 'put':
            self.put_payoffs()
            plt.hist(self.p0, density=0, facecolor='blue', alpha=0.75)
        plt.show()
        return

    def simulated_price(self,position:str = 'put'):
        mean = self.payoffs_mean(position)
        simulated_price = np.exp(-self.r * self.T) * mean
        return simulated_price

    def formulaic_price(self,position:str = 'put'):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if position == 'call':
            formulaic_price = (self.S0 * stats.norm.cdf(d1, 0, 1)
                       - self.K * np.exp(-self.r*self.T) * stats.norm.cdf(d2, 0, 1))
        if position == 'put':
            formulaic_price = (self.K * np.exp(-self.r*self.T) * stats.norm.cdf(-d2, 0, 1)
                       - self.S0 * stats.norm.cdf(-d1, 0, 1))
        return formulaic_price
    
    def price_difference(self,position:str = 'put'):
        sp = self.simulated_price(position)
        print(sp)
        fp = self.formulaic_price(position)
        print(fp)
        delta = abs(sp - fp)
        return delta

class lookback_european_option(european_option):

    def __init__(self,S0=100,r=0,sigma=0.25,T=1,steps=100,K=100,frequency=2000):
        super(lookback_european_option,self).__init__(S0,r,sigma,T,steps,K,frequency)

    def call_payoffs(self):
        self.simulation_paths()
        index_max = np.argmax(self.S, axis=0)
        max_S = self.S[index_max, range(self.S.shape[1])]
        self.c0 = np.maximum(max_S - self.K, 0)
        return self

    def put_payoffs(self):
        self.simulation_paths()
        index_min = np.argmin(self.S, axis=0)
        min_S = self.S[index_min, range(self.S.shape[1])]
        self.p0 = np.maximum(self.K - min_S, 0)
        return self

    def premium(self,position:str = 'put'):
        lookback_sp = self.simulated_price(position)
        S0 = self.S0
        r = self.r
        sigma = self.sigma
        T = self.T
        steps = self.steps
        K = self.K
        frequency = self.frequency
        sp = european_option.simulated_price(european_option(S0,r,sigma,T,steps,K,frequency),position)
        premium = lookback_sp - sp
        return premium

def varying_sigma():
    sigma_list = np.arange(0, 1, 0.05)
    eop_price = []
    lop_price = []
    premium = []

    for i in np.arange(0, 1, 0.05):
        eop = european_option(sigma=i)
        lop = lookback_european_option(sigma=i)
        eop_price.append(eop.simulated_price('put'))
        lop_price.append(lop.simulated_price('put'))
        premium.append(lop.premium('put'))
    list = {'Sigma': sigma_list, 'Euro_option_price': eop_price,
            'Lookback_option_price': lop_price, 'Premium':premium}
    return pd.DataFrame(list)

if __name__ == '__main__':
    #display all columns
    pd.set_option('display.max_columns',None)
    #display all rows
    pd.set_option('display.max_rows', None)

    e = european_option()
    #question a
    e.draw_paths()
    #question b
    print(e.terminal_mean())
    print(e.terminal_var())
    e.draw_payoffs_histogram('put')
    print(e.payoffs_mean('put'))
    print(e.payoffs_std('put'))
    #question c
    print(e.simulated_price('put'))
    #question d
    print(e.price_difference('put'))
    #question e
    f = lookback_european_option()
    # f.put_payoffs()
    # print(np.mean(f.p0))
    # print(f.simulated_price('put'))
    # question f
    premium = f.premium('put')
    print(premium)
    #question g
    print(varying_sigma())