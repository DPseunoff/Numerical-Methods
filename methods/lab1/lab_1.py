import numpy as np
import pandas as pd
import cmath

import methods.lab1.lab_1_1 as l1
import methods.lab1.lab_1_2 as l2
import methods.lab1.lab_1_3 as l3
import methods.lab1.lab_1_4 as l4
import methods.lab1.lab_1_5 as l5


def output(m, round=None):
    if len(m.shape) == 0:
        print(m, '\n')
    else:
        if round is not None:
            m = np.round(m, round)
        print(pd.DataFrame(m).to_string(index=False, header=False), '\n')


def lab1_1():
    a = pd.read_csv('tasks/lab1_a.csv', header=None, sep=' ').to_numpy(dtype=float)
    b = pd.read_csv('tasks/lab1_b.csv', header=None).to_numpy(dtype=float)
    print('Матрица A:')
    output(a)
    print('Правая часть:')
    output(b)
    a = l1.LU(a)
    print('Матрица LU:')
    output(a, 3)
    print('Решение СЛАУ:')
    output(l1.get_SLAU_solve(a, b))
    print('Определитель А: ', end='')
    output(l1.get_determinant(a), 3)
    print('Обратная матрица А:')
    output(l1.get_inverse(a))


def lab1_2():
    a = pd.read_csv('tasks/lab1_2_a.csv', header=None, sep=' ').to_numpy()
    b = pd.read_csv('tasks/lab1_2_b.csv', header=None).to_numpy()
    print('Матрица A:')
    output(a)
    print('Правая часть:')
    output(b)
    if l2.conditions_check(a):
        print('Проверки пройдены\nРешение:')
        output(l2.sweep_method(a, b))
    else:
        print('Проверки не пройдены')


def lab1_3():
    a = pd.read_csv('tasks/lab1_3_a.csv', header=None, sep=' ').to_numpy()
    b = pd.read_csv('tasks/lab1_3_b.csv', header=None).to_numpy()
    print('Матрица A:')
    output(a)
    print('Правая часть:')
    output(b)
    alpha, beta = l3.find_alpha_beta(a, b)
    eps = float(input('Введите эпсилон:\n'))
    print('Решение методом простых итераций:')
    x, k = l3.simple_iterations_method(alpha, beta, eps)
    output(x)
    print('Кол-во проделанных итераций: ', k)
    print('Решение методом Зейделя:')
    x, k = l3.Seidel_method(alpha, beta, eps)
    output(x)
    print('Кол-во проделанных итераций: ', k)


def lab1_4():
    a = pd.read_csv('tasks/lab1_4.csv', header=None, sep=' ').to_numpy()
    n = np.shape(a)[0]
    print('Матрица A:')
    output(a)

    if not l4.check_symmetric(a):
        print('Матрица не симметричная.')
        return

    eps = float(input('Введите эпсилон:\n'))
    u = []
    k = 0
    while l4.return_condition(a, eps):
        i, j = l4.find_max(a)
        u.append(l4.find_U(a, i, j))
        a = u[-1].T.dot(a).dot(u[-1])
        k += 1
    print('Собственные значения: ', [a[i, i] for i in range(n)])
    eigen = [l4.mult_Us(u)[:, j] for j in range(n)]
    print('Собственные векторы: \n')
    for i in range(n):
        print(eigen[i])
    print('Кол-во проделанных итераций: ', k)


def lab1_5():
    a = pd.read_csv('tasks/lab1_5.csv', header=None, sep=' ').to_numpy()
    n = 3
    eps = float(input('Введите эпсилон\n'))
    q, r = l5.QR(a)
    print('Матрица Q:')
    output(q, 2)
    print('Матрица R:')
    output(r, 2)
    eigenvalues = l5.find_eigenvalues(a, eps)
    print("Собственные значения:")
    [print("{:g}".format(eigenvalues[i]), end=" ") for i in range(len(eigenvalues))]


def init():
    while True:
        print('\n---------------------------------')
        print('Выберите номер задания:')
        print('1 - Алгоритм LU')
        print('2 - Метод прогонки')
        print('3 - Метод простых итераций и метод Зейделя')
        print('4 - Метод вращений')
        print('5 - Алгоритм QR')
        print('6 - Назад')
        bl = int(input())
        if bl < 1 or bl > 6:
            print('Неверный номер, введите снова\n')
            continue
        if bl == 6:
            print('"""""""""""""""""""""""""""""""""')
            break
        if bl == 1:
            lab1_1()
            continue
        if bl == 2:
            lab1_2()
            continue
        if bl == 3:
            lab1_3()
            continue
        if bl == 4:
            lab1_4()
            continue
        if bl == 5:
            lab1_5()
            continue
