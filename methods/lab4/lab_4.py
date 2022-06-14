import methods.lab4.lab_4_1 as l1
import methods.lab4.lab_4_2 as l2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def lab4_1():
    borders = (1, 2)
    y0 = 2 + np.exp(1)
    z0 = 1 + np.exp(1)
    h = 0.1

    x = np.arange(borders[0], borders[1] + h, h)
    y = l1.f(x)
    y1 = l1.Euler_method(borders, y0, z0, h)
    y2, z = l1.RungeKutta_method(borders, y0, z0, h)
    y3 = l1.Adams_method(borders, deepcopy(y2), z, h)

    # Получаем решения методов для шага h/2
    h2 = h / 2
    y1_2 = l1.Euler_method(borders, y0, z0, h2)
    y2_2, z2 = l1.RungeKutta_method(borders, y0, z0, h2)
    y3_2 = l1.Adams_method(borders, deepcopy(y2_2), z2, h2)
    print("Оценка погрешности методом Рунге-Ромберга:")
    print("Для явного метода Эйлера:", l1.sqr_error(y1, l1.Runge_Romberg_method(y1, y1_2, h, h2, 1)))
    print("Для метода Рунге-Кутты:", l1.sqr_error(y2, l1.Runge_Romberg_method(y2, y2_2, h, h2, 4)))
    print("Для метода Адамса:", l1.sqr_error(y3, l1.Runge_Romberg_method(y3, y3_2, h, h2, 4)))

    print("\nСравнение с точным решением:")
    print("Для явного метода Эйлера:", l1.sqr_error(y1, y))
    print("Для метода Рунге-Кутты:", l1.sqr_error(y2, y))
    print("Для метода Адамса:", l1.sqr_error(y3, y))

    plt.figure(figsize=(12, 7))
    plt.plot(x, y, label='Точное решение', linewidth=1, color="blue")
    plt.plot(x, y1, label='Явный метод Эйлера', linewidth=1, color="red", ls="--")
    plt.plot(x, y2, label='Метод Рунге-Кутты', linewidth=1, color="yellow", ls="--")
    plt.plot(x, y3, label='Метод Адамса', linewidth=1, color="green", ls="--")
    plt.grid()
    plt.legend()
    plt.show()


def lab4_2():
    # Условие для стрельбы
    ya = 2
    yb = 2.5 - 0.5 * np.log(3)
    borders16 = (0, np.pi / 6)
    h16 = borders16[1]/50
    h16_2 = borders16[1]/100

    x16 = np.arange(borders16[0], borders16[1] + h16, h16)
    y1 = l2.f16(x16)
    y16_1 = l2.shooting_method(l2.ddf16, borders16, ya, yb, h16)
    y16_2 = l2.shooting_method(l2.ddf16, borders16, ya, yb, h16_2)
    print("Оценка погрешности методом Рунге-Ромберга:")
    print(l2.sqr_error(y16_1, l2.Runge_Romberg_method(y16_1, y16_2, h16, h16_2, 1)))
    print("Сравнение с точным решением:")
    print(l2.sqr_error(y16_1, y1))

    # Условие для КР метода
    equation17 = {'p': l2.p17, 'q': l2.q17, 'f': l2.right_f17}
    left_condition17 = {'a': 0, 'b': 1, 'c': 1}
    right_condition17 = {'a': 4, 'b': -1, 'c': 23 * (np.e ** (-4))}
    borders17 = (0, 2)

    h1 = 0.02
    h2 = 0.01
    x17 = np.arange(borders17[0], borders17[1] + h1, h1)
    y2 = l2.f17(x17)
    y17_1 = l2.diff_method(left_condition17, right_condition17, equation17, borders17, h1)
    y17_2 = l2.diff_method(left_condition17, right_condition17, equation17, borders17, h2)
    print("Оценка погрешности методом Рунге-Ромберга:")
    print(l2.sqr_error(y17_1, l2.Runge_Romberg_method(y17_1, y17_2, h1, h2, 1)))
    print("Сравнение с точным решением:")
    print(l2.sqr_error(y17_1, y2))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x16, y1, label='Точное решение', linewidth=1, color="blue")
    ax1.plot(x16, y16_1, label='Метод стрельбы', linewidth=1, color="red", ls="--")
    ax1.set_xlabel('Рис.1. Метод стрельбы')
    ax1.legend()
    ax1.grid()
    ax2.plot(x17, y2, label='Точное решение', linewidth=1, color="blue")
    ax2.plot(x17, y17_1, label='Метод конечных разностей', linewidth=1, color="green", ls="--")
    ax2.set_xlabel('Рис.2. Конечно-разностный метод')
    ax2.legend()
    ax2.grid()
    plt.show()


def init():
    while True:
        print('\n---------------------------------')
        print('Выберите номер задания:')
        print('1 - Методы Эйлера, Рунге-Кутты и Адамса 4-ого порядка')
        print('2 - Метод стрельюы и конечно-разностный метод')
        print('3 - Назад')
        bl = int(input())
        if bl < 1 or bl > 3:
            print('Неверный номер, введите снова\n')
            continue
        if bl == 3:
            print('"""""""""""""""""""""""""""""""""')
            break
        if bl == 1:
            lab4_1()
            continue
        if bl == 2:
            lab4_2()
            continue
