import numpy as np


def f(x):
    return 1 / (256 - x**4)


def ddf(x):
    return (3072 * (x**2) + 20 * (x**6)) / ((256 - x**4)**3)


def rectangles_method(x0, xk, step):
    x = np.arange(x0, xk + step, step)
    n = np.shape(x)[0]
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = (x[i] - x[i - 1])
    summa = 0
    for i in range(1, np.shape(h)[0]):
        summa += h[i] * f((x[i - 1] + x[i]) / 2)

    return summa


def rectangles_error(x0, xk, step):
    M2 = np.max(ddf(np.arange(x0, xk + step, step)))
    r = (xk - x0) * (step ** 2) * M2
    return r / 24


def trapeze_method(x0, xk, step):
    x = np.arange(x0, xk + step, step)
    n = np.shape(x)[0]
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = (x[i] - x[i - 1])
    summa = 0
    for i in range(1, np.shape(h)[0]):
        summa += (f(x[i]) + f(x[i - 1])) * h[i]

    return summa / 2


def trapeze_error(x0, xk, step):
    M2 = np.max(ddf(np.arange(x0, xk + step, step)))
    r = (xk - x0) * (step ** 2) * M2
    return r / 12


def Simpson_method(x0, xk, step):
    x = np.arange(x0, xk + step, step)
    n = np.shape(x)[0]
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = (x[i] - x[i - 1])
    summa = 0
    for i in range(1, np.shape(h)[0]):
        summa += (f(x[i - 1]) + 4 * f((x[i - 1] + x[i]) / 2) + f(x[i])) * (h[i] / 2)

    return summa / 3


def Runge_Romberg_method(f1, f2, h1, h2, p):
    if h1 < h2:
        return f1 + (f1 - f2) / ((h2 / h1) ** p - 1)
    return f2 + (f2 - f1) / ((h1 / h2) ** p - 1)
