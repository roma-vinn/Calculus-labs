#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# by Roman Polishchenko
# 2 course, comp math
# Taras Shevchenko National University of Kyiv
# email: roma.vinn@gmail.com

import numpy as np


def grad_descent(func, der, start_point, eps=10**-6):
    x0 = start_point
    step = 5
    x = x0 - step * der(x0)
    while step > eps:
        if any(np.isinf(x)):
            x = x0
            break
        x = x0 - step * der(x0)
        if func(x) >= func(x0):
            step /= 2
        else:
            x, x0 = x - step*der(x), x
    return x


def newton_method(func, der_1, der_2, start_point, eps=10**-6):
    x0 = start_point
    x = x0 - np.linalg.inv(der_2(x0)).dot(der_1(x0))

    while np.linalg.norm(func(x) - func(x0)) > eps:
        if any(np.isinf(x)):
            x = x0
            break
        x, x0 = x - np.linalg.inv(der_2(x)).dot(der_1(x)), x
    return x


def calc_min(func, func_der, func_der_2, arr_0, neg=False):
    def sign(flag):
        if flag:
            return -1
        else:
            return 1

    def maxmin(flag):
        return 'max' if flag else 'min'

    res = ''
    for x0 in arr_0:
        res += '\nДля точки {}:\n'.format(x0)

        x_min = sign(neg) * grad_descent(func, func_der, x0)
        f_min = sign(neg) * func(x_min)
        res += 'Градієнтний спуск:\n'
        res += 'Порахували: x_{0} = {1}, {0} = {2}\n'.format(maxmin(neg), x_min, f_min)

        res += 'Метод Ньютона:\n'
        x_min_n = sign(neg) * newton_method(func, func_der, func_der_2, x0)
        f_min_n = sign(neg) * func(x_min_n)
        res += 'Порахували: x_{0} = {1}, {0} = {2}\n'.format(maxmin(neg), x_min_n, f_min_n)

        # якщо є модуль scipy, то можна звірити з точними обрахунками
        try:
            from scipy import optimize
            res += 'Точне значення: x_{0} = {1}, {0} = {2}\n'.format(maxmin(neg),
                                                                     sign(neg)*optimize.minimize(func,
                                                                                                 x0,
                                                                                                 method='BFGS').x,
                                                                     sign(neg)*optimize.minimize(func,
                                                                                                 x0,
                                                                                                 method='BFGS').fun)
        except (ImportError, AttributeError):
            pass

    return res


def task_a():
    def func_a(x):
        return 2 * x[0] ** 2 + 3 * x[1] ** 2 - 4 * x[0] + 5 * x[1] - 1

    def func_a_der(x):
        return np.array([4 * x[0] - 4, 6 * x[1] + 5])

    def func_a_der_2(x):
        # + x*0 – дописано тільки для того, щоб PyCharm не сварився на "невикористання x"
        # на виконання алгоритму не впливає жодним чином
        return np.array([[4, 0], [0, 6]]) + x*0

    def neg_func_a(x):
        return func_a(x) * (-1)

    def neg_func_a_der(x):
        return func_a_der(x) * (-1)

    def neg_func_a_der_2(x):
        return func_a_der_2(x) * (-1)

    start_points = np.array([np.array([10, 10])])

    res = ''
    res += '\nМінімуми:\n'
    res += calc_min(func_a, func_a_der, func_a_der_2, start_points)
    res += '\nМаксимуми:\n'
    res += calc_min(neg_func_a, neg_func_a_der, neg_func_a_der_2, start_points, neg=True)
    return res


def task_b():
    def func_b(x):
        return x[0]**2 + 2*x[1]**2 - 4*x[1]**4 - x[0]**4 + 3

    def func_b_der(x):
        return np.array([2*x[0] - 4*x[0]**3, 4*x[1] - 16*x[1]**3])

    def func_b_der_2(x):
        return np.array([[2 - 12*x[0]**2, 0], [0, 4 - 48*x[1]**2]])

    def neg_func_b(x):
        return func_b(x) * (-1)

    def neg_func_b_der(x):
        return func_b_der(x) * (-1)

    def neg_func_b_der_2(x):
        return func_b_der_2(x) * (-1)

    start_points = np.array([[0, 0], [10, 10], [10, -10], [-10, 10], [-10, -10]])

    res = ''
    res += '\nМінімуми:\n'
    res += calc_min(func_b, func_b_der, func_b_der_2, start_points)
    res += '\nМаксимуми:\n'
    res += calc_min(neg_func_b, neg_func_b_der, neg_func_b_der_2, start_points, neg=True)

    return res


def task_c():
    def func_c(x):
        return x[0]**2 - x[1]**2

    def func_c_der(x):
        return np.array([2*x[0], -2*x[1]])

    def func_c_der_2(x):
        return np.array([[2, 0], [0, 2]]) + x*0

    def neg_func_c(x):
        return func_c(x) * (-1)

    def neg_func_c_der(x):
        return func_c_der(x) * (-1)

    def neg_func_c_der_2(x):
        return func_c_der_2(x) * (-1)

    start_points = np.array([[1, 1]])

    res = ''
    res += '\nМінімуми:\n'
    res += calc_min(func_c, func_c_der, func_c_der_2, start_points)
    res += '\nМаксимуми:\n'
    res += calc_min(neg_func_c, neg_func_c_der, neg_func_c_der_2, start_points, neg=True)

    return res


def task_d():
    def func_d(x):
        return (x[0]**2 + x[1]**2 + x[2]**2)**2 +\
               ((x[0] - 2)**2 + (x[1] - 2)**2 + (x[2] - 2)**2)**2

    def func_d_der(x):
        # майже жахіття
        x, y, z = x[0], x[1], x[2]
        return np.array([2 * (x**2 + y**2 + z**2) * (2 * x)
                         + 2 * ((x - 2)**2 + (y - 2)**2 + (z - 2)**2) * (2 * (x - 2)),
                         2 * (x ** 2 + y ** 2 + z ** 2) * (2 * y)
                         + 2 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) * (2 * (y - 2)),
                         2 * (x ** 2 + y ** 2 + z ** 2) * (2 * z)
                         + 2 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) * (2 * (z - 2))])

    def func_d_der_2(x):
        # жахіття
        x, y, z = x[0], x[1], x[2]
        return np.array([[4 * (x ** 2 + y ** 2 + z ** 2) + 4 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) +
                          16 * (x ** 2 - 2 * x + 2),
                          16 * (x * y - x - y + 2),
                          16 * (x * z - x - z + 2)],
                         [16 * (x * y - x - y + 2),
                          4 * (x ** 2 + y ** 2 + z ** 2) + 4 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) +
                          16 * (y ** 2 - 2 * y + 2),
                          16 * (y * z - y - z + 2)],
                         [16 * (x * z - x - z + 2),
                          16 * (y * z - y - z + 2),
                          4 * (x ** 2 + y ** 2 + z ** 2) + 4 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) +
                          16 * (z ** 2 - 2 * z + 2)]])

    def neg_func_d(x):
        return func_d(x) * (-1)

    def neg_func_d_der(x):
        return func_d_der(x) * (-1)

    def neg_func_d_der_2(x):
        return func_d_der_2(x) * (-1)

    start_points = np.array([[3, 3, 3], [-1, -2, -3], [1, 1, 1]])

    res = ''
    res += '\nМінімуми:\n'
    res += calc_min(func_d, func_d_der, func_d_der_2, start_points)
    res += '\nМаксимуми:\n'
    res += calc_min(neg_func_d, neg_func_d_der, neg_func_d_der_2, start_points, neg=True)

    return res


def task_2():
    def func_2(x):
        x, y = x[0], x[1]
        return x**3 + 8*y**3 - 6*x*y + 5

    def func_2_der(x):
        x, y = x[0], x[1]
        return np.array([3*x**2 - 6*y, 24*y**2 - 6*x])

    def func_2_der_2(x):
        x, y = x[0], x[1]
        return np.array([[6*x, -6],
                         [-6, 48*y]])

    def neg_func_2(x):
        return func_2(x) * (-1)

    def neg_func_2_der(x):
        return func_2_der(x) * (-1)

    def neg_func_2_der_2(x):
        return func_2_der_2(x) * (-1)

    start_points = np.array([[1, 1], [1, -1], [2, 1]])

    res = ''
    res += '\nМінімуми:\n'
    res += calc_min(func_2, func_2_der, func_2_der_2, start_points)
    res += '\nМаксимуми:\n'
    res += calc_min(neg_func_2, neg_func_2_der, neg_func_2_der_2, start_points, neg=True)

    return res


if __name__ == '__main__':
    with open('output.txt', 'w') as file:
        print('Завдання 1 a):', file=file)
        print(task_a(), file=file)

        print('Завдання 1 b):', file=file)
        print(task_b(), file=file)

        print('Завдання 1 c):', file=file)
        print(task_c(), file=file)

        print('Завдання 1 d):', file=file)
        print(task_d(), file=file)

        print('Завдання 2:', file=file)
        print('Обрали функцію: x ** 3 + 8 * y ** 3 - 6 * x * y + 5', file=file)
        print(task_2(), file=file)
