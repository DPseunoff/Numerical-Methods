import numpy as np
from methods.lab1.lab_1_1 import get_inverse, LU


def f(x):
    return np.array([[float(3*x[0] - np.cos(x[1]))], [float(3*x[1]-np.exp(x[0]))]])


def phi(x):
    return np.array([[float(np.cos(x[1])/3)], [float(np.exp(x[0])/3)]])


def Jacobi(x):
    return np.array([[3, float(np.sin(x[1]))], [float(-np.exp([0])), 3]])


def phiJacobi(x):
    return np.array([[0, float(-np.sin(x[1])/3)], [float(np.exp(x[0])/2), 0]])


def norm(a):
    return np.max(np.sum(abs(a), axis=1))


def while_condition(x_cur, x_prev, eps):
    return abs(x_cur[0] - x_prev[0]) > eps or abs(x_cur[1] - x_prev[1]) > eps


def Newton_method(x0, eps):
    x_prev = x0
    x_cur = x_prev - np.dot(get_inverse(LU(Jacobi(x_prev))), f(x_prev))
    k = 0
    while while_condition(x_cur, x_prev, eps):
        x_prev = x_cur
        x_cur = x_prev - np.dot(get_inverse(LU(Jacobi(x_prev))), f(x_prev))
        k += 1
    return x_cur, k


def iterations_method(x0, eps, phi):
    x_prev = x0
    k = 0
    x_cur = phi(x_prev)
    if norm(phiJacobi(x_prev)) < 1:
        print("Условие сходимости не выполняется")
        return None, None
    while while_condition(x_cur, x_prev, eps):
        x_prev = x_cur
        x_cur = phi(x_prev)
        k += 1
    return x_cur, k
