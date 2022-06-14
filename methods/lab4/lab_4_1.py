from copy import deepcopy
import numpy as np


def ddf(x, func, df):
    return ((x + 1) * df - func) / x


def f(x):
    return x + 1 + np.exp(x)


def Euler_method(borders, y0, z0, h):
    x = np.arange(borders[0], borders[1] + h, h)
    n = np.shape(x)[0]
    y = np.zeros(n)
    z = np.zeros(n)
    y[0] = y0
    z[0] = z0

    for i in range(0, n - 1):
        z[i + 1] = z[i] + h * ddf(x[i], y[i], z[i])
        y[i + 1] = y[i] + h * z[i]

    return y


def RungeKutta_method(borders, y0, z0, h):
    x = np.arange(borders[0], borders[1] + h, h)
    n = np.shape(x)[0]
    y = np.zeros(n)
    z = np.zeros(n)
    y[0] = y0
    z[0] = z0

    for i in range(n - 1):
        k1 = h * z[i]
        l1 = h * ddf(x[i], y[i], z[i])
        k2 = h * (z[i] + 0.5 * l1)
        l2 = h * ddf(x[i] + 0.5 * h, y[i] + 0.5 * k1, z[i] + 0.5 * l1)
        k3 = h * (z[i] + 0.5 * l2)
        l3 = h * ddf(x[i] + 0.5 * h, y[i] + 0.5 * k2, z[i] + 0.5 * l2)
        k4 = h * (z[i] + l3)
        l4 = h * ddf(x[i] + h, y[i] + k3, z[i] + l3)
        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        delta_z = (l1 + 2 * l2 + 2 * l3 + l4) / 6
        y[i + 1] = y[i] + delta_y
        z[i + 1] = z[i] + delta_z

    return y, z


def Adams_method(borders, calc_y, calc_z, h):
    x = np.arange(borders[0], borders[1] + h, h)
    n = np.shape(x)[0]
    y = calc_y[:4]
    z = calc_z[:4]
    for i in range(3, n - 1):
        # предиктор
        z_next = z[i] + (h / 24) * (
                55 * ddf(x[i], y[i], z[i]) - 59 * ddf(x[i - 1], y[i - 1], z[i - 1]) + 37 * ddf(x[i - 2], y[i - 2],
                                                                                               z[i - 2]) - 9 * ddf(
            x[i - 3], y[i - 3], z[i - 3]))
        y_next = y[i] + (h / 24) * (55 * z[i] - 59 * z[i - 1] + 37 * z[i - 2] - 9 * z[i - 3])

        # корректор
        z_i = z[i] + (h / 24) * (9 * ddf(x[i + 1], y_next, z_next) +
                                 19 * ddf(x[i], y[i], z[i]) -
                                 5 * ddf(x[i - 1], y[i - 1], z[i - 1]) +
                                 1 * ddf(x[i - 2], y[i - 2], z[i - 2]))
        z = np.append(z, z_i)

        y_i = y[i] + (h / 24) * (9 * z_next + 19 * z[i] - 5 * z[i - 1] + 1 * z[i - 2])
        y = np.append(y, y_i)

    return y


def sqr_error(y, y_correct):
    return np.sqrt(np.sum((y - y_correct) ** 2))


def Runge_Romberg_method(y1, y2, h1, h2, p):
    if h1 > h2:
        k = int(h1 / h2)
        y = np.zeros(np.shape(y1)[0])
        for i in range(np.shape(y1)[0]):
            y[i] = y2[i * k] + (y2[i * k] - y1[i]) / (k ** p - 1)
        return y
    else:
        k = int(h2 / h1)
        y = np.zeros(np.shape(y2)[0])
        for i in range(np.shape(y2)[0]):
            y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (k ** p - 1)
        return y
