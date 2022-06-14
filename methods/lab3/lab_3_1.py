import copy
import numpy as np


def f(x):
    return np.exp(x) + x


def df(x):
    return np.exp(x)


def Lagrange_polynomial(x, X, f):
    n = np.shape(X)[0]
    summa = 0
    for i in range(n):
        composition = 1

        # Вычисление лагранжевых многочленов влияния
        for j in range(n):
            if i != j:
                composition *= (x-X[j])/(X[i]-X[j])

        summa += f[i] * composition
    return summa


def Lagrange_error(x, X):
    n = np.shape(X)[0]
    w = 1
    for i in range(n):
        w *= (x - X[i])
    M2 = np.max(abs(df(X)))
    factorial = np.math.factorial(n)
    return (M2 * w) / factorial


# Разделенные разности
def find_diffs(X, n, f):
    diff = [copy.copy(f)]
    for i in range(1, n):
        tmp = []
        for k in range(n - i):
            tmp.append((diff[i - 1][k] - diff[i - 1][k + 1]) / (X[k] - X[k + i]))
        diff.append(tmp)
    return diff


def Newton_polynomial(x, X, f):
    n = np.shape(X)[0]
    diffs = find_diffs(X, n, f)
    summa = f[0]

    for i in range(1, n):
        composition = 1
        for j in range(i):
            composition *= x - X[j]
        summa += composition * diffs[i][0]

    return summa


def Newton_error(x, X, f):
    n = np.shape(X)[0]
    diffs1 = find_diffs(X, n - 1, f)
    diffs2 = find_diffs(X, n, f)
    prev = f[0]
    curr = f[0]

    for i in range(1, n - 1):
        composition = 1
        for j in range(i):
            composition *= x - X[j]
        prev += composition * diffs1[i][0]

    for i in range(1, n):
        composition = 1
        for j in range(i):
            composition *= x - X[j]
        curr += composition * diffs2[i][0]

    return abs(curr - prev)
