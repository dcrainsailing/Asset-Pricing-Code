from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def bisection(fun, start, end, precision):

    s = fun.subs(f, start)
    e = fun.subs(f, end)
    if e * s > 0:
        print
        "No solution!"
        return False
    if s == 0:
        print
        "Solution is ", start
        return start
    if e == 0:
        print
        "Solution is ", end
        return end
    while abs(end - start) > precision:
        mid = (start + end) / 2.0
        m = fun.subs(f, mid)
        if m == 0:
            print
            "Solution is ", mid
            return mid
        if s * m < 0:
            end = mid
        if m * e < 0:
            start = mid
    print
    "Solution is ", start
    return start


def bootstrapping(rate, Term, frequency):
    sigma = frequency
    numerator = 0
    denominator = 0
    i = 0.5
    j = 0.5
    while i <= Term:
        index = (round(i + 0.1) - 1)
        numerator = 1 - exp(-sum(rate[0:index]) - rate[index])

        if (rate[index] != f):
            if(i%1 == 0):
                loc_ = 1
            else:
                loc_ = i%1
            denominator = denominator + sigma * exp(-sum(rate[0:index])) * exp(-rate[index] * loc_)
            i = i + frequency
        elif (rate[index] == f):
            loc = rate.index(f)
            denominator = denominator + sigma * exp(-sum(rate[0:loc])) * exp(-rate[index] * j)
            j = j + frequency
            i = i + frequency
    # print(numerator)
    # print(denominator)
    # forward_rate = solve(numerator / denominator - Swap_rate / 100)

    return numerator / denominator

def forward_rate(rate, Swap_rate, Term, frequency):
    funct = bootstrapping(rate, Term, frequency) - Swap_rate / 100
    forward_rate = bisection(funct, Swap_rate / 100 - 1, Swap_rate / 100 + 1,
                             0.0000001)
    return forward_rate

def no_percentage(x):
    return x/100

if __name__ == '__main__':
    f = Symbol('f')
    rate = [f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f]
    swap_rate = [2.8438, 3.060, 3.126, 3.144, 3.150, 3.169, 3.210, 3.237]
    Terms = [1, 2, 3, 4, 5, 7, 10, 30]
    rate[0] = forward_rate(rate, swap_rate[0], Terms[0], 0.5)
    rate[1] = forward_rate(rate, swap_rate[1], Terms[1], 0.5)
    rate[2] = forward_rate(rate, swap_rate[2], Terms[2], 0.5)
    rate[3] = forward_rate(rate, swap_rate[3], Terms[3], 0.5)
    rate[4] = forward_rate(rate, swap_rate[4], Terms[4], 0.5)
    rate_5to7 = forward_rate(rate, swap_rate[5], Terms[5], 0.5)
    rate[5] = rate_5to7
    rate[6] = rate_5to7
    rate_7to10 = forward_rate(rate, swap_rate[6], Terms[6], 0.5)
    rate[7] = rate_7to10
    rate[8] = rate_7to10
    rate[9] = rate_7to10
    rate_10to30 = forward_rate(rate, swap_rate[7], Terms[7], 0.5)
    rate[10] = rate_10to30
    Forward_rate = rate[0:6] + rate[7:8] + rate[10:11]
    print(Forward_rate)

    Swap_rate = list(map(no_percentage, swap_rate))
    plt.plot(Terms, Swap_rate)
    plt.plot(Terms, Forward_rate)
    plt.show()

    rate[10] = f
    Swap_0to15 = bootstrapping(rate, 15, 0.5).subs(f,rate_10to30)
    print(Swap_0to15)

    gap = np.array(Terms) - np.array([0] + Terms[:-1])
    discount_factor = []
    zero_rate = []
    weighted_forward = gap * np.array(Forward_rate)
    k = 1
    while k <= 8:
        D = exp(-sum(weighted_forward[0:k]))
        zero = -log(D)/Terms[k-1]
        discount_factor = discount_factor + [D]
        zero_rate = zero_rate + [zero]
        k = k + 1

    print('discount_factor:')
    print(discount_factor)
    print('zero_rate:')
    print(zero_rate)
    plt.plot(Terms, Forward_rate)
    plt.plot(Terms,Swap_rate)
    plt.plot(Terms,zero_rate)
    plt.show()

    new_swaprate = []
    new_fowardrate = [f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f]
    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[0], 0.5).subs(f, (Forward_rate[0] + 100 * 0.0001))]
    new_fowardrate[0] = (Forward_rate[0] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[1], 0.5).subs(f, (Forward_rate[1] + 100 * 0.0001))]
    new_fowardrate[1] = (Forward_rate[1] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[2], 0.5).subs(f, (Forward_rate[2] + 100 * 0.0001))]
    new_fowardrate[2] = (Forward_rate[2] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[3], 0.5).subs(f, (Forward_rate[3] + 100 * 0.0001))]
    new_fowardrate[3] = (Forward_rate[3] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[4], 0.5).subs(f, (Forward_rate[4] + 100 * 0.0001))]
    new_fowardrate[4] = (Forward_rate[4] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[5], 0.5).subs(f, (Forward_rate[5] + 100 * 0.0001))]
    new_fowardrate[5] = (Forward_rate[5] + 100 * 0.0001)
    new_fowardrate[6] = (Forward_rate[5] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[6], 0.5).subs(f, (Forward_rate[6] + 100 * 0.0001))]
    new_fowardrate[7] = (Forward_rate[6] + 100 * 0.0001)
    new_fowardrate[8] = (Forward_rate[6] + 100 * 0.0001)
    new_fowardrate[9] = (Forward_rate[6] + 100 * 0.0001)

    new_swaprate = new_swaprate + [bootstrapping(new_fowardrate, Terms[7], 0.5).subs(f, (Forward_rate[7] + 100 * 0.0001))]
    new_fowardrate = new_fowardrate[0:5] + new_fowardrate[5:6] + new_fowardrate[7:8] + [Forward_rate[7] + 100 * 0.0001]
    print('new_swaprate:')
    print(new_swaprate)
    plt.plot(Terms, new_swaprate)
    plt.plot(Terms, Swap_rate)
    plt.show()

    bearish_shift = [0,0,0,0.05,0.1,0.15,0.25,0.50]
    bull_shift = [-0.50, -0.25, -0.15, -0.1, -0.05, 0, 0, 0]
    bearish_swap_rate = []
    bull_swap_rate = []
    id = 0
    while(id < len(swap_rate)):
        bearish_swap_rate = bearish_swap_rate + [swap_rate[id] + bearish_shift[id]]
        bull_swap_rate = bull_swap_rate + [swap_rate[id] + bull_shift[id]]
        id = id + 1

    bearish_rate = [f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f]
    bearish_rate[0] = forward_rate(bearish_rate, bearish_swap_rate[0], Terms[0], 0.5)
    bearish_rate[1] = forward_rate(bearish_rate, bearish_swap_rate[1], Terms[1], 0.5)
    bearish_rate[2] = forward_rate(bearish_rate, bearish_swap_rate[2], Terms[2], 0.5)
    bearish_rate[3] = forward_rate(bearish_rate, bearish_swap_rate[3], Terms[3], 0.5)
    bearish_rate[4] = forward_rate(bearish_rate, bearish_swap_rate[4], Terms[4], 0.5)
    rate_5to7 = forward_rate(bearish_rate, bearish_swap_rate[5], Terms[5], 0.5)
    bearish_rate[5] = rate_5to7
    bearish_rate[6] = rate_5to7
    rate_7to10 = forward_rate(bearish_rate, bearish_swap_rate[6], Terms[6], 0.5)
    bearish_rate[7] = rate_7to10
    bearish_rate[8] = rate_7to10
    bearish_rate[9] = rate_7to10
    rate_10to30 = forward_rate(bearish_rate, bearish_swap_rate[7], Terms[7], 0.5)
    bearish_rate[10] = rate_10to30
    bearish_Forward_rate = bearish_rate[0:6] + bearish_rate[7:8] + bearish_rate[10:11]
    print('bearish_swap_rate')
    print(bearish_swap_rate)
    print('bearish_Forward_rate')
    print(bearish_Forward_rate)
    plt.plot(Terms,list(map(no_percentage,bearish_swap_rate)))
    plt.plot(Terms,bearish_Forward_rate)
    plt.show()
    plt.plot(Terms, Forward_rate)
    plt.plot(Terms, bearish_Forward_rate)
    plt.show()

    bull_rate = [f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f]
    bull_rate[0] = forward_rate(bull_rate, bull_swap_rate[0], Terms[0], 0.5)
    bull_rate[1] = forward_rate(bull_rate, bull_swap_rate[1], Terms[1], 0.5)
    bull_rate[2] = forward_rate(bull_rate, bull_swap_rate[2], Terms[2], 0.5)
    bull_rate[3] = forward_rate(bull_rate, bull_swap_rate[3], Terms[3], 0.5)
    bull_rate[4] = forward_rate(bull_rate, bull_swap_rate[4], Terms[4], 0.5)
    rate_5to7 = forward_rate(bull_rate, bull_swap_rate[5], Terms[5], 0.5)
    bull_rate[5] = rate_5to7
    bull_rate[6] = rate_5to7
    rate_7to10 = forward_rate(bull_rate, bull_swap_rate[6], Terms[6], 0.5)
    bull_rate[7] = rate_7to10
    bull_rate[8] = rate_7to10
    bull_rate[9] = rate_7to10
    rate_10to30 = forward_rate(bull_rate, bull_swap_rate[7], Terms[7], 0.5)
    bull_rate[10] = rate_10to30
    bull_Forward_rate = bull_rate[0:6] + bull_rate[7:8] + bull_rate[10:11]
    print('bull_swap_rate')
    print(bull_swap_rate)
    print('bull_Forward_rate')
    print(bull_Forward_rate)
    plt.plot(Terms,list(map(no_percentage,bull_swap_rate)))
    plt.plot(Terms,bull_Forward_rate)
    plt.show()
    plt.plot(Terms, Forward_rate)
    plt.plot(Terms, bull_Forward_rate)
    plt.show()