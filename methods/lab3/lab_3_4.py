import numpy as np


def F(x):
    return np.sin(x)+np.log(x)


def differentiation(x, X, f):
    n = np.shape(X)[0]

    # Проверка на принадлежность к одному из интервалов
    for i in range(0, n - 2):
        if x == X[i]:
            # Формула 5
            d1 = (3 * f[i] - 4 * f[i + 1] + f[i + 2]) / 2 * (X[i + 2] - X[i + 1])
            # Формула 6
            d2 = (f[i] - 2 * f[i + 1] + f[i + 2]) / (X[i + 2] - X[i + 1]) ** 2

        if X[i] < x < X[i + 1]:
            # Разделенные разности 1-ого и 2-ого порядка
            diff1_i1 = (f[i + 1] - f[i]) / (X[i + 1] - X[i])
            diff1_i2 = (f[i + 2] - f[i + 1]) / (X[i + 2] - X[i + 1])
            diff2 = (diff1_i2 - diff1_i1) / (X[i + 2] - X[i + 1])

            d1 = diff1_i1 + diff2 * (2 * x - X[i] - X[i + 1])
            d2 = 2 * diff2

    return d1, d2
