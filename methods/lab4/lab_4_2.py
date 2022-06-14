import numpy as np
import matplotlib.pyplot as plt
from decimal import *


# Вторая производная, выраженная через x, f, f'
def ddf16(x, f, df):
    return np.tan(x) * df - 2 * f


# Точные решения
def f16(x):
    return np.sin(x) + 2 - np.sin(x) * np.log((1 + np.sin(x)) / (1 - np.sin(x)))


def f17(x):
    return (1 + x) * np.exp(-x * x)


# Функция перед y'
def p16(x):
    return -np.tan(x)


def p17(x):
    return 4 * x


# Функция перед y
def q16(x):
    return 2


def q17(x):
    return 4 * x * x + 2


# Функция в правой части уравнения
def right_f16(x):
    return 0


def right_f17(x):
    return 0


def shooting_method(ddy, borders, ya, yb, h):
    y0 = ya
    eta1 = 0.5
    eta2 = 2.0
    resolve1 = Runge_Kutty_method(ddy, borders, y0, eta1, h)[0]
    resolve2 = Runge_Kutty_method(ddy, borders, y0, eta2, h)[0]
    Phi1 = resolve1[-1] - yb
    Phi2 = resolve2[-1] - yb
    while abs(Phi2 - Phi1) > h/10:
        temp = eta2
        eta2 = eta2 - (eta2 - eta1) / (Phi2 - Phi1) * Phi2
        eta1 = temp
        resolve1 = Runge_Kutty_method(ddy, borders, y0, eta1, h)[0]
        resolve2 = Runge_Kutty_method(ddy, borders, y0, eta2, h)[0]
        Phi1 = resolve1[-1] - yb
        Phi2 = resolve2[-1] - yb

    return Runge_Kutty_method(ddy, borders, y0, eta2, h)[0]


def diff_method(BCondition1, BCondition2, equation, borders, h):
    x = np.arange(borders[0], borders[1] + h, h)
    N = np.shape(x)[0]

    a1 = BCondition1['a']; b1 = BCondition1['b']; c1 = BCondition1['c']
    a2 = BCondition2['a']; b2 = BCondition2['b']; c2 = BCondition2['c']

    # Составляем СЛАУ
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0][0] = -(3 * h / 2) + (2 - h * equation['p'](x[1])) * h / (4 + 2 * h * equation['p'](x[1]))
    A[0][1] = 2 * h + (h * h * equation['q'](x[1]) - 2) * h / (2 + h * equation['p'](x[1]))
    A[0][2] = 0
    b[0] = c1 * h * h

    for i in range(1, N-1):
        A[i][i-1] = 1 - h * equation['p'](x[i]) / 2
        A[i][i] = -2 + h * h * equation['q'](x[i])
        A[i][i+1] = 1 + h * equation['p'](x[i]) / 2
        b[i] = equation['f'](x[i])
    A[N - 1][N - 3] = 0
    A[N - 1][N - 2] = -2 * b2 * h  - ((h * h * h * equation['q'](x[N - 2]) - 2 * h) * b2) / (2 - h * equation['p'](x[N - 2]))
    A[N - 1][N - 1] = h * h * a2 + 3 * h * b2 / 2 - ((2 * h + h * h * equation['p'](x[N - 2])) * b2) / ((2 - h * equation['p'](x[N - 2])) * 2)
    b[N - 1] = c2 * h * h
    return Solve(A, b)


# Метод Рунге-Кутты решения задачи Коши
def Runge_Kutty_method(ddy, borders, y0, z0, h):
    x = np.arange(borders[0], borders[1] + h, h)
    N = np.shape(x)[0]
    y = np.zeros(N)
    z = np.zeros(N)
    y[0] = y0; z[0] = z0

    for i in range(N-1):
        K1 = h * z[i]
        L1 = h * ddy(x[i], y[i], z[i])
        K2 = h * (z[i] + 0.5 * L1)
        L2 = h * ddy(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * (z[i] + 0.5 * L2)
        L3 = h * ddy(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * (z[i] + L3)
        L4 = h * ddy(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y[i+1] = y[i] + delta_y
        z[i+1] = z[i] + delta_z

    return y, z


# Метод прогонки
def Solve(A, b):
    p = np.zeros(len(b))
    q = np.zeros(len(b))

    # Прямой ход: поиск прогоночных коэффициентов P и Q
    p[0] = -A[0][1]/A[0][0]
    q[0] = b[0]/A[0][0]
    for i in range(1, len(p)-1):
        p[i] = -A[i][i+1]/(A[i][i] + A[i][i-1]*p[i-1])
        q[i] = (b[i] - A[i][i-1]*q[i-1])/(A[i][i] + A[i][i-1]*p[i-1])

    p[-1] = 0
    q[-1] = (b[-1] - A[-1][-2]*q[-2])/(A[-1][-1] + A[-1][-2]*p[-2])

    # Обратный ход: поиск x
    x = np.zeros(len(b))
    x[-1] = q[-1]
    for i in reversed(range(len(b)-1)):
        x[i] = p[i]*x[i+1] + q[i]
    return x


def sqr_error(y, y_correct):
    return np.sqrt(np.sum((y-y_correct)**2))


def Runge_Romberg_method(y1, y2, h1, h2, p):
    if h1 > h2:
        k = int(h1 / h2)
        y = np.zeros(np.shape(y1)[0])
        for i in range(np.shape(y1)[0]):
            y[i] = y2[i*k]+(y2[i*k]-y1[i])/(k**p-1)
        return y
    else:
        k = int(h2 / h1)
        y = np.zeros(np.shape(y2)[0])
        for i in range(np.shape(y2)[0]):
            y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (k ** p - 1)
        return y
