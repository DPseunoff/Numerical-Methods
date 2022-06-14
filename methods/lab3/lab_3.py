import methods.lab3.lab_3_1 as l1
import methods.lab3.lab_3_2 as l2
import methods.lab3.lab_3_3 as l3
import methods.lab3.lab_3_4 as l4
import methods.lab3.lab_3_5 as l5
import numpy as np
import matplotlib.pyplot as plt


def lab3_1():
    x1 = np.array([-2, -1, 0, 1])
    f1 = l1.f(x1)
    x2 = np.array([-2, -1, 0.2, 1])
    f2 = l1.f(x2)
    our_x = -0.5

    print(f"Разница в точке x = {round(our_x, 3)}:"
          f"\nДля Лагранжа: {abs(l1.f(our_x) - l1.Lagrange_polynomial(our_x, x1, f1))}"
          f"\nОшибка Лагранжа: {l1.Lagrange_error(our_x, x1)}"
          f"\nДля Ньютона: {abs(l1.f(our_x) - l1.Newton_polynomial(our_x, x1, f1))}"
          f"\nОшибка Ньютона: {l1.Newton_error(our_x, x1, f1)}")
    x = np.arange(-2, 1, 0.01)

    plt.figure(figsize=(12, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(x1, f1, color="black", s=25)
    plt.plot(x, l1.f(x), label='f(x)', linewidth=1, color="blue")
    plt.plot(x, [float(l1.Lagrange_polynomial(xi, x1, f1)) for xi in x], label='Многочлен Лагранжа', linewidth=1,
             color="blue")
    plt.plot(x, [float(l1.Newton_polynomial(xi, x1, f1)) for xi in x], label='Многочлен Ньютона', linewidth=1,
             color="red")
    plt.title("Пункт а")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(x2, f2, color="black", s=25)
    plt.plot(x, l1.f(x), label='f(x)', linewidth=1, color="blue")
    plt.plot(x, [float(l1.Lagrange_polynomial(xi, x2, f2)) for xi in x], label='Многочлен Лагранжа', linewidth=1,
             color="blue")
    plt.plot(x, [float(l1.Newton_polynomial(xi, x2, f2)) for xi in x], label='Многочлен Ньютона', linewidth=1,
             color="red")
    plt.title("Пункт б")
    plt.legend()
    plt.grid()
    plt.show()


def lab3_2():
    xi = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    fi = np.array([-1.8647, -0.63212, 1.0, 3.7183, 9.3891])
    a, b, c, d = l2.find_coefficients(xi, fi)

    print(f"Значение в точке x = -0.5: {l2.spline(-0.5, xi, a, b, c, d)}")
    x = np.arange(-2.0, 2.0, 0.01)
    plt.figure(figsize=(12, 7))

    plt.scatter(xi, fi, color="blue", s=25)
    plt.plot(x, [float(l2.spline(xk, xi, a, b, c, d)) for xk in x], linewidth=1, color="red")
    plt.grid()
    plt.show()


def lab3_3():
    xi = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    fi = np.array([-2.9502, -1.8647, -0.63212, 1, 3.7183, 9.3891])

    a1, b1 = l3.MNK_1(xi, fi)
    a2, b2, c2 = l3.MNK_2(xi, fi)
    a3, b3, c3, d3 = l3.MNK_3(xi, fi)
    a4, b4, c4, d4, e4 = l3.MNK_4(xi, fi)

    x = np.arange(-3.0, 2.0, 0.01)
    print("Сумма квадратов ошибок:")
    print("Многочлен 1-ой степени:", l3.find_errors(fi, np.array([float(a1 + b1 * xk) for xk in xi])))
    print("Многочлен 2-ой степени:",
          l3.find_errors(fi, np.array([float(a2 + b2 * xk + c2 * xk * xk) for xk in xi])))
    print("Многочлен 3-ой степени:",
          l3.find_errors(fi, np.array([float(a3 + b3 * xk + c3 * xk * xk + d3 * xk * xk * xk) for xk in xi])))
    plt.figure(figsize=(12, 7))
    plt.scatter(xi, fi, color="black", s=25)
    plt.plot(x, [float(a1 + b1 * xk) for xk in x], label='Многочлен 1-ой степени', linewidth=1, color="red")
    plt.plot(x, [float(a2 + b2 * xk + c2 * xk * xk) for xk in x], label='Многочлен 2-ой степени', linewidth=1,
             color="blue")
    plt.plot(x, [float(a3 + b3 * xk + c3 * xk * xk + d3 * xk * xk * xk) for xk in x])
    plt.plot(x, [float(a4 + b4 * xk + c4 * xk * xk + d4 * xk * xk * xk + e4 * xk * xk * xk * xk) for xk in x])

    plt.grid()
    plt.legend()
    plt.show()


def lab3_4():
    xi = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
    fi = np.array([-0.40136, 0, 0.40136, 0.81152, 1.2435])
    print("Значение 1-й производной в точке x = 0.2:", l4.differentiation(0.2, xi, fi)[0])
    print("Значение 2-й производной в точке x = 0.2:", l4.differentiation(0.2, xi, fi)[1])


def lab3_5():
    x0 = -2
    xk = 2
    h1 = 1.0
    h2 = 0.5
    rect1 = l5.rectangles_method(x0, xk, h1)
    rect2 = l5.rectangles_method(x0, xk, h2)
    rect_runge = l5.Runge_Romberg_method(rect1, rect2, h1, h2, 1)
    rect_error1 = l5.rectangles_error(x0, xk, h1)
    rect_error2 = l5.rectangles_error(x0, xk, h2)
    print("Метод прямоугольников:")
    print(f"h1 = {h1}: F = {rect1}")
    print(f"h2 = {h2}: F = {rect2}")
    print(f"Уточнение методом Рунге-Ромберга-Ричардсона: F = {rect_runge}")
    print("Ошибки: ", rect_error1, "и", rect_error2)


    trapeze1 = l5.trapeze_method(x0, xk, h1)
    trapeze2 = l5.trapeze_method(x0, xk, h2)
    trapeze_runge = l5.Runge_Romberg_method(trapeze1, trapeze2, h1, h2, 2)
    trapeze_error1 = l5.trapeze_error(x0, xk, h1)
    trapeze_error2 = l5.trapeze_error(x0, xk, h2)
    print("Метод трапеций:")
    print(f"h1 = {h1}: F = {trapeze1}")
    print(f"h2 = {h2}: F = {trapeze2}")
    print(f"Уточнение методом Рунге-Ромберга-Ричардсона: F = {trapeze_runge}")
    print(f"Погрешности: {abs(trapeze_runge - trapeze1)} и {abs(trapeze_runge - trapeze2)}")
    print("Ошибки: ", trapeze_error1, "и", trapeze_error2)

    simpson1 = l5.Simpson_method(x0, xk, h1)
    simpson2 = l5.Simpson_method(x0, xk, h2)
    simpson_runge = l5.Runge_Romberg_method(simpson1, simpson2, h1, h2, 4)
    print("Метод Симпсона:")
    print(f"h1 = {h1}: F = {simpson1}")
    print(f"h2 = {h2}: F = {simpson2}")
    print(f"Уточнение методом Рунге-Ромберга-Ричардсона: F = {simpson_runge}")
    print(f"Погрешности: {abs(simpson_runge - simpson1)} и {abs(simpson_runge - simpson2)}")


def init():
    while True:
        print('\n---------------------------------')
        print('Выберите номер задания:')
        print('1 - Интерполяционные многочлены Лагранжа и Ньютона')
        print('2 - Кубический сплайн')
        print('3 - МНК')
        print('4 - Производные')
        print('5 - Определенный интеграл')
        print('6 - Назад')
        bl = int(input())
        if bl < 1 or bl > 6:
            print('Неверный номер, введите снова\n')
            continue
        if bl == 6:
            print('"""""""""""""""""""""""""""""""""')
            break
        if bl == 1:
            lab3_1()
            continue
        if bl == 2:
            lab3_2()
            continue
        if bl == 3:
            lab3_3()
            continue
        if bl == 4:
            lab3_4()
            continue
        if bl == 5:
            lab3_5()
            continue
