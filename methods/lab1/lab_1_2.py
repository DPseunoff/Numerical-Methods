import numpy as np


def sweep_method(a, b):
    p = np.zeros(len(b))
    q = np.zeros(len(b))

    # прямой ход
    p[0] = -a[0][1]/a[0][0]
    q[0] = b[0]/a[0][0]
    for i in range(1, len(p)-1):
        p[i] = -a[i][i+1]/(a[i][i] + a[i][i-1]*p[i-1])
        q[i] = (b[i] - a[i][i-1]*q[i-1])/(a[i][i] + a[i][i-1]*p[i-1])
    p[-1] = 0
    q[-1] = (b[-1] - a[-1][-2]*q[-2])/(a[-1][-1] + a[-1][-2]*p[-2])

    # обратный ход
    x = np.zeros(len(b))
    x[-1] = q[-1]
    for i in reversed(range(len(b)-1)):
        x[i] = p[i]*x[i+1] + q[i]

    return x


def conditions_check(a):

    # проверка на квадратную матрицу
    if np.shape(a)[0] != np.shape(a)[1]:
        return False

    # проверка на преобладание диагональных коэффициентов
    n = np.shape(a)[0]
    for i in range(1, n - 1):
        if abs(a[i][i]) < abs(a[i][i - 1]) + abs(a[i][i + 1]):
            return False

    # проверка на |с1 / b1| < 1 и |aN / bN| < 1
    if abs(a[0][0]) <= abs(a[0][1]) or abs(a[n - 1][n - 1]) <= abs(a[n - 1][n - 2]):
        return False

    return True
