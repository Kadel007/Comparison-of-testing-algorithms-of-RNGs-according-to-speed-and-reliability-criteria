import numpy as np
import time as tm
import matplotlib.pyplot as plt
import math as mt


def f_theoretical(x):
    n = len(x)
    f = np.zeros(n)
    a = 1 / n
    for i in range(n):
        f[i] = i * a
    return f


def f_theor_summ(x):
    n = len(x)
    f = np.zeros(n)
    for i in range(n):
        if (x[i] >= 0) & (x[i] <= 1.0):
            f[i] = 0.5 * x[i] * x[i]
        elif (x[i] > 1.0) & (x[i] <= 2):
            f[i] = -1.0 + 2 * x[i] - 0.5 * x[i] * x[i]
        else:
            f[i] = 1
    return f


def edf_summ(z, x):
    n = len(x)
    m = len(z)
    f = np.zeros(n)
    for k in range(m):
        for i in range(n):
            if (z[k] <= x[i]):
                f[i] += 1
    f = f / m
    return f


# empirical distribution function for full test
def edf(z):
    n = len(z)
    f = np.zeros(n)
    f[0] = z[0]
    for k in range(1, n):
        f[k] = f[k - 1] + z[k]
    coeff = 1 / n
    f = coeff * f
    return f


# Kolmogorov criterion
def Kolm(p):
    d = -0.5 * np.log((1 - p) / 2)
    d = np.sqrt(d)
    return d


def inv_kolm(d):
    p = 1 - 2 * mt.exp(-2 * d * d)
    return p


if __name__ == '__main__':
    # global const, grid
    N = 100  # the number of columns and rows in the frequency matrix
    M = N * N  # the number of values of the empirical distribution function
    Q = 10  # the average number of hits in a cell of the probability table
    K = Q * M  # number of random numbers in 2 sequences
    E = 10  # number of experiments
    alpha = 0.15  # correlation coefficient

    print(f"---------- M= {M:0.0f}, N= {N:0.0f} E= {E:0.0f} K= {K:0.0f} ----------")

    x = np.arange(M) / M  # array of arguments

    tfr = f_theoretical(x)  # theoretical distribution function

    # graphics preparation
    plt.figure()
    plt.rc('font', size=9)
    plt.rcParams['font.family'] = 'Times New Roman'

    # ----------------------------------------------
    print('for full test -------------------------')

    N1 = int(K * alpha)
    d = np.zeros(E)
    experience = 0
    t1 = tm.perf_counter()
    plt.subplot(2, 1, 1)
    while experience < E:
        y1 = N * np.random.rand(K)
        y2 = N * np.random.rand(K)

        for i in range(N1):  # artificial simulation of dependence
            y2[i] = y1[i]

        z = np.zeros(M)  # array of EDF values
        for i in range(K):
            n = mt.trunc(y1[i])
            m = mt.trunc(y2[i])
            k = m * N + n
            z[k] += 1
        # print(m, n, k, z[k])    # debugging

        efr = edf(z) / Q
        dd = abs(efr - tfr)

        d[experience] = np.max(dd)

        if experience == 0:
            plt.plot(x, efr, 'b-', linewidth=0.5, label='експериметральні ФР') 
            plt.plot(x, tfr, 'r-', linewidth=1.0, label='теоретична ФР F(z)')
            plt.legend()
        else:
            plt.plot(x, efr, 'b-', linewidth=0.5)
        experience += 1

    D = np.mean(d) * np.sqrt(K)
    p = inv_kolm(D)
    t2 = tm.perf_counter()
    dt = t2 - t1

    print(f"Вибірка N = {K:0.0f}, Коефіцієнт кореляції  R = {alpha:0.3f}")
    print(f"Функція Колмогорова К = {D:0.4f}, Час t = {dt:0.3f} секунд, Ймовірність залежності P0 = {p:0.5f}")

    D = np.mean(d) * np.sqrt(K)
    p = inv_kolm(D)
    t2 = tm.perf_counter()

    plt.grid(True)

    # --------------------------------------
    print('for summ test -------------------------')
    # K = K
    # K = mt.trunc(mt.sqrt(K))
    K = Q * N
    N1 = int(K * alpha)
    d = np.zeros(E)
    # dx = 0.01; x0 = 0; xm = 2
    dx = 1 / N
    x0 = 0
    xm = 2
    # x=np.arange(x0, xm+dx, dx)
    x = np.arange(2 * N) / N
    tfr = f_theor_summ(x)
    # print(len(tfr), len(x))
    nx = len(x)
    experience = 0
    t1 = tm.perf_counter()
    plt.subplot(2, 1, 2)
    while experience < E:
        y1 = np.random.rand(K)
        y2 = np.random.rand(K)

        for i in range(N1):  # artificial simulation of dependence
            y2[i] = y1[i]

        y = (2 * N * (y1 + y2))
        z = np.zeros(2 * N)  # array of EDF values

        for i in range(K):
            k = mt.trunc((y[i]) / 2)
            z[k] += 1

        efr = 2 * edf(z) * N / K
        dd = abs(efr - tfr)

        d[experience] = np.max(dd)

        if experience == 0:
            plt.plot(x, efr, 'b-', linewidth=0.5, label='експериметральні ФР') 
            plt.plot(x, tfr, 'r-', linewidth=1.0, label='теоретична ФР F(z)')
            plt.legend()
        else:
            plt.plot(x, efr, 'b-', linewidth=0.5)
        experience += 1

    D = np.mean(d) * np.sqrt(K)
    p = inv_kolm(D)
    t2 = tm.perf_counter()
    dt = t2 - t1

    print(f"Вибірка N = {K:0.0f}, Коефіцієнт кореляції  R = {alpha:0.3f}")
    print(f"Функція Колмогорова К = {D:0.4f}, Час t = {dt:0.3f} секунд, Ймовірність залежності P0 = {p:0.5f}")

    plt.grid(True)
    plt.show()
