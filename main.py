import methods.lab1.lab_1 as lab1
import methods.lab2.lab_2 as lab2
import methods.lab3.lab_3 as lab3
import methods.lab4.lab_4 as lab4


def main():
    while True:
        print('Введите соответствующий номер:')
        print('1 - Методы решения задач линейной алгебры')
        print('2 - Методы решения нелинейных уравнений и систем нелинейных уравнений')
        print('3 - Методы приближения функций. Численное дифференцирование и интегрирование')
        print('4 - Методы решения начальных и краевых задач для обыкновенных дифференциальных уравнений (ОДУ) и систем '
              'ОДУ')
        print('5 - Выход из программы')
        bl = int(input())
        if bl < 1 or bl > 5:
            print('Неверный номер, введите снова\n')
            continue
        elif bl == 5:
            print('Работа программы завершена завершена')
            break
        if bl == 1:
            lab1.init()
            continue
        if bl == 2:
            lab2.init()
            continue
        if bl == 3:
            lab3.init()
            continue
        if bl == 4:
            lab4.init()
            continue


main()
