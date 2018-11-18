from math import cos, log, atan
import numpy as np
import random
import sympy

# Задача 1


def func_1(x):
    return 0.5*cos(x)


def task_1():
    eps = 1e-9
    lamb = 0.5
    x0 = 0
    x1 = func_1(x0)
    d = abs(x0-x1)
    m = int(log((eps*(1-lamb) / d), lamb)) + 1
    for i in range(m):
        x1 = func_1(x1)
    return x1


with open('output.txt', 'w') as file:
    print('Task 1:', file=file)
    print('Ans =\n{}\n'.format(task_1()), file=file)

# Задача 2


def func_2(x):
    return x**5 - x - 1


def func_2_diff(x):
    return 5*x**4 - 1


def task_2():
    a = 1
    b = 2
    eps = 1e-9
    m = func_2_diff(a)
    M = func_2_diff(b)
    lamb_1 = 1 / (2 * M)

    def f(x):
        return x - lamb_1*func_2(x)

    lamb = 1 - m/(2*M)
    x0 = 1.5
    x1 = f(x0)
    d = abs(x0-x1)
    m = int(log((eps*(1-lamb) / d), lamb)) + 1
    for i in range(m):
        x1 = f(x1)
    return x1


with open('output.txt', 'a') as file:
    print('Task 2:', file=file)
    print('Ans =\n{}\n'.format(task_2()), file=file)

# Задача 3


def rand_matrix(size: tuple, a: float, b: float, unique=True) -> np.ndarray:
    n = size[0]*size[1]
    if unique:
        res = np.array([random.uniform(a, b) for _ in range(n)]).reshape(size)
    else:
        ran = random.uniform(a, b)
        res = np.array([ran] * n).reshape(size)
    return res


def find_lamb(matrix):
    return np.linalg.norm(matrix)


def task_3_a():
    eps = 10**-9
    m = 10
    c = rand_matrix((m, m), -0.05, 0.05)
    b = rand_matrix((m, 1), -10, 10)
    lamb = find_lamb(c)
    x0 = np.array([0]*m).reshape(m, 1)
    a = c - np.eye(m)

    def u(x):
        return a.dot(x) - b + x

    x1 = u(x0)
    d = abs(np.linalg.norm(x1-x0))
    n = int(log((eps * (1 - lamb) / d), lamb)) + 1

    for i in range(n):
        x1 = u(x1)

    return x1


with open('output.txt', 'a') as file:
    print('Task 3 (a):', file=file)
    print('Ans =\n{}\n'.format(task_3_a()), file=file)


def task_3_b():
    eps = 10**-9
    m = 80
    c = rand_matrix((m, m), -0.01, 0.01, unique=False)
    b = rand_matrix((m, 1), -10, 10)
    lamb = find_lamb(c)
    x0 = np.array([0]*m).reshape(m, 1)
    a = c - np.eye(m)

    def u(x):
        return a.dot(x) - b + x

    x1 = u(x0)
    d = abs(np.linalg.norm(x1-x0))
    n = int(log((eps * (1 - lamb) / d), lamb)) + 1

    for i in range(n):
        x1 = u(x1)

    return x1


with open('output.txt', 'a') as file:
    print('Task 3 (b):', file=file)
    print('Ans =\n{}'.format(task_3_b()), file=file)


# def func_4(t):
#     return 0.5*atan(y(t))

def task_4():
    # from lab1.lab_1_integrate import integral_simpson
    def integral_simpson(func, a, b, n=1000):
        """
        :param func: f(x)
        :param a: lower bound
        :param b: upper bound
        :param n: number of intervals
        :return: float
        """
        m = (b - a) / n
        res = 0
        for k in range(0, n):
            xk1 = a + m * k
            xk2 = a + m * (k + 1)
            res += 1 / 6 * (func(xk1) + 4 * func(xk1 / 2 + xk2 / 2) + func(xk2)) * (xk2 - xk1)

        return res


    eps = 1e-9
    y0 = 1
    a = 0
    b = 1
    def f(x, y):
        return y0 + integral_simpson()