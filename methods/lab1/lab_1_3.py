import numpy as np


# нормы
def norm1(a):
    return np.max(abs(a))


# Евклидова норма
def norm2(a):
    return (np.sum(a**2))**0.5


# проверка норм
def norm_check(a, eps):
    return norm1(a) > eps or norm2(a) > eps


# проверка на диагональное преобладание
def diagonal_check(a):

    if np.shape(a)[0] != np.shape(a)[1]:
        return False

    row_check = True
    column_check = True
    n = np.shape(a)[0]

    for i in range(n):
        summa = 0
        for j in range(n):
            if i != j:
                summa += abs(a[i][j])
        if abs(a[i][i]) < summa:
            row_check = False

    for j in range(n):
        summa = 0
        for i in range(n):
            if i != j:
                summa += abs(a[i][j])
        if abs(a[j][j]) < summa:
            column_check = False

    return column_check or row_check


def find_alpha_beta(a, b):
    n = a.shape[0]
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        alpha[i] = -a[i] / a[i][i]
        beta[i] = b[i]/a[i][i]
        alpha[i][i] = 0
    return alpha, beta


def simple_iterations_method(alpha, beta, eps):
    prev = beta
    curr = beta + np.dot(alpha, beta)
    i = 0
    while norm_check(curr - prev, eps):
        prev = curr
        curr = beta + np.dot(alpha, prev)
        i += 1
    return curr, i


def new_vector(prev, alpha, beta):
    n = np.shape(alpha)[0]
    curr = np.copy(prev)
    for i in range(n):
        summa = 0
        for j in range(n):
            summa += alpha[i, j] * curr[j]
        curr[i] = beta[i] + summa
    return curr


def Seidel_method(alpha, beta, eps):
    prev = beta
    curr = new_vector(prev, alpha, beta)
    i = 0
    while norm_check(curr - prev, eps):
        prev = curr
        curr = new_vector(prev, alpha, beta)
        i += 1
    return curr, i
