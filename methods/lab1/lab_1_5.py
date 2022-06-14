import numpy as np
import cmath


# Евклидова норма
def norm(a):
    return (np.sum(a ** 2)) ** 0.5


# Разбиение матрицы на блоки размером 2x2
def create_blocks(a):
    n = np.shape(a)[0]
    if n % 2 == 1:
        blocks = [[0 for i in range(n // 2 + 1)] for j in range(n // 2 + 1)]
    else:
        blocks = [[0 for i in range(n // 2)] for j in range(n // 2)]

    for i in range(0, n - 1, 2):
        for j in range(0, n - 1, 2):
            blocks[i // 2][j // 2] = np.array([[a[i][j], a[i][j + 1]], [a[i + 1][j], a[i + 1][j + 1]]])

    if n % 2 == 1:
        for i in range(0, n - 1, 2):
            blocks[i // 2][-1] = np.array([[a[i][-1]], [a[i + 1][-1]]])
        for j in range(0, n - 1, 2):
            blocks[-1][j // 2] = np.array([a[-1][j], a[-1][j + 1]])
        blocks[-1][-1] = np.array(a[-1][-1])

    return blocks


# Решение квадратного уравнения
def get_solve(a):
    d = (a[0][0]+a[1][1])**2-4*(a[0][0]*a[1][1] - a[0][1]*a[1][0])
    d_sqrt = cmath.sqrt(d)
    return (a[0][0]+a[1][1]+d_sqrt)/2, (a[0][0]+a[1][1]-d_sqrt)/2


def condition_1(a, eps):
    n = np.shape(a)[0]
    sum = 0
    for i in range(1, n):
        for j in range(i):
            sum += a[i][j]**2

    return sum**0.5 > eps


def condition_2(a, eps):
    blocks = create_blocks(a)
    summa = 0
    for i in range(1, len(blocks)):
        for j in range(i):
            summa += np.sum(blocks[i][j] ** 2)
    return summa ** 0.5 > eps


# QR-разложение матрицы A
def QR(A):
    n = np.shape(A)[0]
    e = np.array([0 for i in range(n)])
    l = np.array([1 for i in range(n)])
    E = np.eye(n)
    r = A.copy()
    q = np.eye(n)
    for i in range(n-1):
        e[i] = 1
        v = (r[:, i] * l + np.sign(r[i, i])*norm(r[:, i] * l)*e).reshape(n, 1)
        H = E - (2 / v.T.dot(v)) * v.dot(v.T)
        q = q.dot(H)
        r = H.dot(r)
        e[i] = 0
        l[i] = 0

    return q, r


def find_eigenvalues(A, eps):
    n = np.shape(A)[0]
    curr = A
    while condition_1(curr,  eps) and condition_2(curr, eps):
        prev = curr
        q, r = QR(prev)
        curr = r.dot(q)

    if not condition_2(curr, eps):
        return [curr[i][i] for i in range(n)]
    else:
        blocks = create_blocks(curr)
        if n % 2 == 0:
            ans = []
            for i in range(len(blocks)):
                s = get_solve(blocks[i][i])
                ans.append(s[0])
                ans.append(s[1])
            return ans
        else:
            ans = []
            for i in range(len(blocks)-1):
                s = get_solve(blocks[i][i])
                ans.append(s[0])
                ans.append(s[1])
            ans.append(blocks[-1][-1])
            return ans
