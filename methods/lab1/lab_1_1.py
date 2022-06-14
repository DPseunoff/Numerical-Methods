import numpy as np


# ЛУ разложение
def LU(a):
    n = a.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            c = a[j][i] / a[i][i]
            a[j][i] = c
            for k in range(i + 1, n):
                a[j][k] = a[j][k] - a[i][k] * c
    return a


# решение СЛАУ
def get_SLAU_solve(a, b):
    n = a.shape[0]
    x = np.zeros(n)
    z = np.zeros(n)

    # Решение Lz = b
    def sum_lz(i):
        res = 0
        for j in range(i):
            res += a[i, j] * z[j]
        return res

    for i in range(n):
        z[i] = b[i] - sum_lz(i)

    # Решение Ux = z
    def sum_ux(i):
        res = 0
        for j in range(i, n):
            res += a[i, j] * x[j]
        return res

    for i in reversed(range(n)):
        x[i] = (z[i] - sum_ux(i)) / a[i, i]
    return x


# получение обратной матрицы
def get_inverse(a):
    n = a.shape[0]
    lower_matrix = np.tril(a, -1) + np.eye(n)
    upper_matrix = np.triu(a)
    x = np.zeros((n, n))
    e = []
    y = []
    for i in range(n):
        e.append(np.zeros(n))
        e[i][i] = 1
        y.append(get_SLAU_solve(lower_matrix, e[i]))
        x[i] = get_SLAU_solve(upper_matrix, y[i])
    return x.transpose()


# получение определителя
def get_determinant(a):
    d = 1
    for i in range(a.shape[0]):
        d *= a[i, i]
    return d
