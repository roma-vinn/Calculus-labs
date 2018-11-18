from math import cos, sin, exp, log2
from random import uniform


def f1(x):
    # проміжок не був вказаний, тому я використав [0, 1]
    return x + cos(x)


def f2(x):
    # [0.9, 1]
    return 3**(-x**2)


def f3(x):
    # [0, 1]
    return sin(x**10)


def my_func(x):
    # [0.5, 1]
    return exp(x)**2 + log2(x)**3


def integral_rectangle(func, a, b, n=1000):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param n: number of intervals
    :return: float
    """
    m = (b-a)/n
    res = 0
    for k in range(0, n):
        xk1 = a + m * k
        xk2 = a + m * (k + 1)
        res += func((xk1 + xk2)/2) * (xk2-xk1)

    return res


def integral_trapeze(func, a, b, n=1000):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param n: number of intervals
    :return: float
    """
    m = (b-a)/n
    res = 0
    for k in range(0, n):
        xk1 = a + m * k
        xk2 = a + m * (k + 1)
        res += (func(xk1) + func(xk2))/2 * (xk2-xk1)

    return res


def integral_simpson(func, a, b, n=1000):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param n: number of intervals
    :return: float
    """
    m = (b-a)/n
    res = 0
    for k in range(0, n):
        xk1 = a + m * k
        xk2 = a + m * (k + 1)
        res += 1/6 * (func(xk1) + 4*func(xk1/2 + xk2/2) + func(xk2)) * (xk2-xk1)

    return res


def integral_carlo(func, a, b, n=1000):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param n: number of intervals
    :return: float
    """
    res = 0
    for k in range(0, n):
        res += func(uniform(a, b))
    res = res * (b-a) / n
    return res


with open('output_integrate.txt', 'w') as file:
    print('НАБЛИЖЕНЕ IНТЕГРУВАННЯ', file=file)

    print('\nФункція f1(x) = x + cos(x) на відрізку [0, 1].', file=file)
    print('Метод прямокутників:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_rectangle(f1, 0, 1, i)), file=file)
    print('Метод трапецій:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_trapeze(f1, 0, 1, i)), file=file)
    print('Метод Симпсона:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_simpson(f1, 0, 1, i)), file=file)
    print('Метод Монте-Карло:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_carlo(f1, 0, 1, i)), file=file)

    print('\nФункція f2(x) = 3**(-x**2) на відрізку [0.9, 1].', file=file)
    print('Метод прямокутників:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_rectangle(f2, 0.9, 1, i)), file=file)
    print('Метод трапецій:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_trapeze(f2, 0.9, 1, i)), file=file)
    print('Метод Симпсона:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_simpson(f2, 0.9, 1, i)), file=file)
    print('Метод Монте-Карло:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_carlo(f2, 0.9, 1, i)), file=file)

    print('\nФункція f3(x) = sin(x**10) на відрізку [0, 1].', file=file)
    print('Метод прямокутників:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_rectangle(f3, 0, 1, i)), file=file)
    print('Метод трапецій:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_trapeze(f3, 0, 1, i)), file=file)
    print('Метод Симпсона:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_simpson(f3, 0, 1, i)), file=file)
    print('Метод Монте-Карло:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_carlo(f3, 0, 1, i)), file=file)

    print('\nФункція my_func(x) = exp(x)**2 + log2(x)**3 на відрізку [0.5, 1].', file=file)
    print('Метод прямокутників:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_rectangle(my_func, 0.5, 1, i)), file=file)
    print('Метод трапецій:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_trapeze(my_func, 0.5, 1, i)), file=file)
    print('Метод Симпсона:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_simpson(my_func, 0.5, 1, i)), file=file)
    print('Метод Монте-Карло:', file=file)
    for i in [100, 10000, 1000000]:
        print('\tn = {}: {}'.format(i, integral_carlo(my_func, 0.5, 1, i)), file=file)
