#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# by Roman Polishchenko
# 2 course, comp math
# Taras Shevchenko National University of Kyiv
# email: roma.vinn@gmail.com

import numpy as np


def check_accuracy(func, x: np.ndarray, x0: np.ndarray, n=1000):
    delta = np.array((x - x0) / n)
    return np.array([np.linalg.norm(func(x0 + i * delta) - func(x0))
                     for i in range(n)]).max() * np.linalg.norm(x - x0)


def task_1():
    def func_1(_x):
        return _x[0]**5 * np.log(1 + _x[1])

    def func_1_der(_x):
        return np.array([5 * _x[0]**4 * np.log(1 + _x[1]), _x[0]**5 / (1 + _x[1])])

    x0 = np.array([1, 0])
    xs = [np.array([1.5, 0.7]), np.array([1.05, 0.07]), np.array([1.005, 0.007])]
    res = ''
    for x in xs:
        tmp = func_1(x0) + func_1_der(x0).dot(x - x0)
        res += ('~ {}, = {}, acc = {}\n'.format(tmp,
                                                func_1(x),
                                                check_accuracy(func_1_der, x, x0)))

    return res


def task_2():
    def func_2(_x):
        return np.e**(_x[0] + _x[1] + _x[2])

    def func_2_der(_x):
        return np.array([func_2(_x)]*3)

    res = ''
    x0 = np.array([0, 0, 0])
    x1 = np.array([0.1, 0.05, -0.01])
    tmp = func_2(x0) + func_2_der(x0).dot(x1 - x0)
    res += ('~ {}, = {}, acc = {}\n'.format(tmp,
                                            func_2(x1),
                                            check_accuracy(func_2_der, x1, x0)))
    eps = 0.1
    delta = 10**-3
    x = x0
    i = 1
    while check_accuracy(func_2_der, x, x0) < eps:
        x = x0 + i * delta
        i += 1

    res += '∂-окіл: ∂ = {}.\n'.format(x)
    return res


def task_3():
    def func_3(_x1, _x2):
        return _x1**2 + _x2**4

    def func_3_der(_x1, _x2):
        return np.array([2*_x1, 4*_x2**3])

    def func_res(_x1, _x2):
        _x0 = np.array([1, 1])
        # tmp = func_3_der(_x0[0], _x0[1])
        # tmp2 = (np.array([_x1, _x2]) - _x0).transpose()

        f = func_3(_x0[0], _x0[1]) + \
            func_3_der(_x0[0], _x0[1])[0]*(_x1 - _x0[0]) + \
            func_3_der(_x0[0], _x0[1])[1] * (_x2 - _x0[1])
        return f

    def make_data():
        # Строим сетку в интервале от -10 до 10 с шагом 0.1 по обоим координатам
        _x = np.arange(-3, 3, 0.1)
        _y = np.arange(-3, 3, 0.1)

        # Создаем двумерную матрицу-сетку
        x_grid, y_grid = np.meshgrid(_x, _y)

        # В узлах рассчитываем значение функции
        return x_grid, y_grid

    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    x, y = make_data()
    z1 = func_3(x, y)

    z2 = func_res(x, y)

    fig = pylab.figure()
    axes = Axes3D(fig)

    axes.plot_surface(x, y, z1)
    axes.plot_surface(x, y, z2)
    pylab.show()


def task_4():
    def func_4(_x):
        return (_x[0] - _x[1] + 1)*(np.sin(_x[0] + _x[1]))

    def func_4_der_1(_x):
        """Перша похідна."""
        return np.array([np.cos(_x[0] + _x[1])*(_x[0] - _x[1] + 1) + np.sin(_x[0] + _x[1]),
                         np.cos(_x[0] + _x[1])*(_x[0] - _x[1] + 1) - np.sin(_x[0] + _x[1])])

    def func_4_der_2(_x):
        """Друга похідна."""
        return np.array([[2*np.cos(_x[0] + _x[1]) - np.sin(_x[0] + _x[1])*(_x[0] - _x[1] + 1),
                          np.sin(_x[0] + _x[1])*(_x[1] - _x[0] - 1)],
                         [np.sin(_x[0] + _x[1])*(_x[1] - _x[0] - 1),
                          np.sin(_x[0] + _x[1])*(_x[1] - _x[0] - 1) - 2*np.cos(_x[0] + _x[1])]])

    x = np.array([0.1, 0.05])
    # х0 оберемо близьким до значення х
    x0 = np.array([0.05, 0.02])

    # при n = 1
    res1 = func_4(x0) + (x - x0).dot(func_4_der_1(x0))
    # при n = 2
    res2 = res1 + (x - x0).transpose().dot(func_4_der_2(x0).dot(x - x0))/2

    res = 'Точний результат: {}\nПри n=1: {}\nПри n=2: {}'.format(func_4(x), res1, res2)
    return res


if __name__ == '__main__':

    print('Завдання 1:')
    print(task_1())

    print('Завдання 2:')
    print(task_2())

    print('Завдання 3:')
    task_3()

    print('Завдання 4:')
    print(task_4())
