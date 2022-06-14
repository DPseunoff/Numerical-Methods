import numpy as np


def return_condition(a, eps):
    res = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != j:
                res += a[i, j]**2
    return res**0.5 > eps


def find_max(a):
    res = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if i != j and abs(a[i, j]) > res:
                res = abs(a[i, j])
                res_i = i
                res_j = j
    return res_i, res_j


def find_U(a, i, j):
    u = np.eye(a.shape[0])
    phi = np.pi/4 if a[i, i] == a[j, j] else 0.5*np.arctan((2*a[i, j])/(a[i, i] - a[j, j]))
    u[i, j] = -np.sin(phi)
    u[j, i] = np.sin(phi)
    u[i, i] = u[j, j] = np.cos(phi)
    return u


def mult_Us(u):
    res = u[0]
    for i in range(1, len(u)):
        res = res.dot(u[i])
    return res


def check_symmetric(a):
    if np.array_equal(a, a.T):
        return True
    else:
        return False
