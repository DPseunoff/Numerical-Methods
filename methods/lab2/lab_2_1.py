import numpy as np


def f(x):
    return 4**x - 5 * x - 2


def df(x):
    return np.log(4) * (4**x) - 5


def ddf(x):
    return (np.log(4)**2) * (4**x)


# Итерирующая функция для положительного корня
def p_phi(x):
    return np.log(5*x + 2) / np.log(4)


def p_diff_phi(x):
    return 5 / (np.log(4) * (5 * x + 2))


# Итерирующая функция для отрицательного корня
def n_phi(x):
    return (4**x - 2) / 5


def n_diff_phi(x):
    return (np.log(4) * (4**x)) / 5


# Достаточное условие сходимости для итерации
def iterations_check(interval, step, func):
    for i in np.arange(interval[0], interval[1], step):
        if abs(func(i)) > 1:
            return False
    return True


# Достаточное условие сходимости для Ньютона
def Newton_check(interval, step, f, df, ddf):
    for i in np.arange(interval[0], interval[1], step):
        if abs(f(i) * ddf(i)) >= df(i) ** 2:
            return False
    return True


def Newton_method(interval, eps, f, df, ddf):
    if f(interval[0]) * ddf(interval[0]) > 0:
        x0 = interval[0]
    else:
        x0 = interval[1]

    x1 = x0 - f(x0) / df(x0)
    k = 0
    while abs(x1 - x0) >= eps:
        x0 = x1
        x1 = x0 - f(x0) / df(x0)
        k += 1
    return x1, k


def iterations_method(interval, eps, phi):
    x0 = (interval[1] - interval[0]) / 2
    x1 = phi(x0)
    k = 0
    while abs(x1 - x0) > eps:
        x0 = x1
        x1 = phi(x0)
        k += 1
    return x1, k


# Проверка на наличие корня
def check_root_existence(interval):
    a = interval[0]
    b = interval[1]
    if np.sign(f(a) * f(b)) > 0:
        return False
    return True
