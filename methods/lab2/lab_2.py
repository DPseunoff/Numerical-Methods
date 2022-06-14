import methods.lab2.lab_2_1 as l1
import methods.lab2.lab_2_2 as l2
import numpy as np


def lab2_1():
    print("Заданное уравнение: 4^x - 5x - 2 = 0")
    eps = float(input('Введите эпсилон: '))
    a = float(input('Введите левый край: '))
    b = float(input('Введите правый край: '))
    interval = [a, b]
    print('Заданный интервал для отрицательного корня: ', interval)

    if l1.check_root_existence(interval):
        print('На данном интервале есть корень')
    else:
        print('На данном интервале корня нет')
        return

    print("Метод Ньютона:")
    if l1.Newton_check(interval, 0.01, l1.f, l1.df, l1.ddf):
        x, k = l1.Newton_method(interval, eps, l1.f, l1.df, l1.ddf)
        print('Решение: ', round(x, 3))
        print("Достигнуто на итерации: ", k)
    else:
        print("Заданный интервал не подходит для данного метода.")
    print("Методы простых итераций:")
    if l1.iterations_check(interval, 0.01, l1.n_diff_phi):
        x, k = l1.iterations_method(interval, eps, l1.n_phi)
        print("Решение: ", round(x, 3))
        print("Достигнуто на итерации: ", k)
    else:
        print("Заданный интервал не подходит для данного метода.")

    a = float(input('Введите левый край: '))
    b = float(input('Введите правый край: '))
    interval = [a, b]
    print('Заданный интервал для отрицательного корня: ', interval)

    if l1.check_root_existence(interval):
        print('На данном интервале есть корень')
    else:
        print('На данном интервале корня нет')
        return

    print("Метод Ньютона:")
    if l1.Newton_check(interval, 0.01, l1.f, l1.df, l1.ddf):
        x, k = l1.Newton_method(interval, eps, l1.f, l1.df, l1.ddf)
        print('Решение: ', round(x, 3))
        print("Достигнуто на итерации: ", k)
    else:
        print("Заданный интервал не подходит для данного метода.")

    print("Методы простых итераций:")
    if l1.iterations_check(interval, 0.01, l1.p_diff_phi):
        x, k = l1.iterations_method(interval, eps, l1.p_phi)
        print("Решение: ", round(x, 3))
        print("Достигнуто на итерации: ", k)
    else:
        print("Заданный интервал не подходит для данного метода.")


def lab2_2():
    print("Заданная система:\n3x1^2 - cos(x2) = 0\n3x2 - e^(x1) = 0")
    x0 = np.array([[1], [1]])
    eps = float(input('Введите эпсилон: '))
    print("\nМетод Ньютона:")
    x, k = l2.Newton_method(x0, eps)
    print(f"Решение: x1 = {round(float(x[0]), 3)}; x2 = {round(float(x[1]), 3)}")
    print("Достигнуто на итерации: ", k)
    x1 = np.array([[1], [1]])
    print("\nМетод простой итерации:")
    x, k = l2.iterations_method(x1, eps, l2.phi)
    if not x is None:
        print(f"Решение: x1 = {round(float(x[0]), 3)}; x2 = {round(float(x[1]), 3)}")
    print("Достигнуто на итерации: ", k)


def init():
    while True:
        print('\n---------------------------------')
        print('Выберите номер задания:')
        print('1 - Решение нелинейных уравнений')
        print('2 - Решение систем нелинейных уравнений')
        print('3 - Назад')
        bl = int(input())
        if bl < 1 or bl > 3:
            print('Неверный номер, введите снова\n')
            continue
        if bl == 3:
            print('"""""""""""""""""""""""""""""""""')
            break
        if bl == 1:
            lab2_1()
            continue
        if bl == 2:
            lab2_2()
            continue
