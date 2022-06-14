import numpy as np
from methods.lab1.lab_1_1 import get_SLAU_solve, LU


# 1-ая степень
def MNK_1(x, f):
    n = np.shape(x)[0]

    a = np.array([[n, np.sum(x)],
                 [np.sum(x), np.sum(x*x)]])
    b = np.array([[np.sum(f)],
                  [np.sum(x*f)]])

    return get_SLAU_solve(LU(a), b)


# 2-ая степень
def MNK_2(x, f):
    n = np.shape(x)[0]

    a = np.array([[n, np.sum(x), np.sum(x * x)],
                  [np.sum(x), np.sum(x * x), np.sum(x * x * x)],
                  [np.sum(x * x), np.sum(x * x * x), np.sum(x * x * x * x)]])
    b = np.array([[np.sum(f)],
                  [np.sum(x * f)],
                  [np.sum(x * x * f)]])

    return get_SLAU_solve(LU(a), b)


def MNK_3(x, f):
    n = np.shape(x)[0]

    a = np.array([[n, np.sum(x), np.sum(x * x), np.sum(x * x * x)],
                  [np.sum(x), np.sum(x * x), np.sum(x * x * x), np.sum(x * x * x * x)],
                  [np.sum(x * x), np.sum(x * x * x), np.sum(x * x * x * x), np.sum(x * x * x * x * x)],
                  [np.sum(x * x * x), np.sum(x * x * x * x), np.sum(x * x * x * x * x),  np.sum(x * x * x * x * x * x)]
                  ])
    b = np.array([[np.sum(f)], [np.sum(x * f)], [np.sum(x * x * f)], [np.sum(x * x * x * f)]])
    return get_SLAU_solve(LU(a), b)


def MNK_4(x, f):
    n = np.shape(x)[0]

    a = np.array([[n, np.sum(x), np.sum(x * x), np.sum(x * x * x), np.sum(x * x * x * x)],
                  [np.sum(x), np.sum(x * x), np.sum(x * x * x), np.sum(x * x * x * x), np.sum(x * x * x * x * x)],
                  [np.sum(x * x), np.sum(x * x * x), np.sum(x * x * x * x), np.sum(x * x * x * x * x), np.sum(x**6)],
                  [np.sum(x * x * x), np.sum(x * x * x * x), np.sum(x * x * x * x * x), np.sum(x**6), np.sum(x**7)],
                  [np.sum(x * x * x * x), np.sum(x * x * x * x * x), np.sum(x**6), np.sum(x**7), np.sum(x**8)]
                  ])
    b = np.array([[np.sum(f)],
                  [np.sum(x * f)],
                  [np.sum(x * x * f)],
                  [np.sum(x * x * x * f)],
                  [np.sum(x * x * x * x * f)]
                  ])
    return get_SLAU_solve(LU(a), b)


# Среднее квадратичное отклонение
def find_errors(f, p):
    return np.sqrt(np.sum((p - f)**2))
