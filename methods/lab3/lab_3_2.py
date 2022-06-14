import numpy as np
from methods.lab1.lab_1_2 import sweep_method


def find_coefficients(X, f):
    n = np.shape(X)[0]

    # Поиск разностей h
    h = [0]
    for i in range(1, n):
        h.append(X[i] - X[i - 1])

    # Поиск коэффициентов c
    c = np.array([0, 0])

    # Заполняем систему для решения методом прогонки
    A = np.zeros((n - 2, n - 2))
    A[0][0] = 2 * (h[1] + h[2])
    A[0][1] = h[2]
    for i in range(3, n - 1):
        A[i - 2][i - 3] = h[i - 1]
        A[i - 2][i - 2] = 2 * (h[i - 1] + h[i])
        A[i - 2][i - 1] = h[i]
    A[n - 3][n - 4] = h[n - 2]
    A[n - 3][n - 3] = 2 * (h[n - 2] + h[n - 1])

    b = np.zeros((n - 2, 1))
    for i in range(2, n):
        b[i - 2] = 3 * (((f[i] - f[i - 1]) / h[i]) - ((f[i - 1] - f[i - 2]) / h[i - 1]))

    # Решаем систему
    c = np.hstack([c, sweep_method(A, b)])

    # Поиск коэффициентов a
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = f[i - 1]

    # Поиск коэффициентов b, d
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(1, n - 1):
        b[i] = (f[i] - f[i - 1]) / h[i] - (h[i] * (c[i + 1] + 2 * c[i])) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    b[n - 1] = (f[n - 1] - f[n - 2]) / h[n - 1] - (2 * h[n - 1] * c[n - 1]) / 3
    d[n - 1] = -c[n - 1] / (3 * h[n - 1])

    return a, b, c, d


def spline(x, X, a, b, c, d):
    n = np.shape(X)[0]

    # Проверка на принадлежность к одному из инервалов
    i = -1
    for k in range(1, n):
        if X[k - 1] <= x < X[k]:
            i = k
    # Нахождение значения кубического сплайна для найденного интервала
    diff = x - X[i - 1]
    return float(a[i] + b[i] * diff + c[i] * diff * diff + d[i] * diff * diff * diff)