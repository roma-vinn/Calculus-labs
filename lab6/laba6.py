import math
import numpy as np
import scipy.integrate


def uniform_expansion(a, b, n):
    rez = []
    for i in range(n + 1):
        rez.append(a + ((b - a) / n) * i)
    return rez


def uniform_expansion_1(a, b, n):
    ch = []
    for i in range(n + 1):
        ch.append(a + ((b - a) / n ** 2) * i ** 2)
    return ch


def uniform_expansion_2(a, b, n):
    h = []
    for i in range(n + 1):
        h.append(b - ((b - a)*(n - i) ** 2) / n ** 2)
    return h


def trap(f, n, sp):
    res = 0
    for i in range(n):
        res += f((sp[i] + sp[i + 1]) / 2) * (sp[i + 1] - sp[i])
    return res


def fun_1(eps, a, n):
    fun = lambda x: math.e ** (-x ** 2)
    A = math.log(1 / eps, math.e)
    tmp = uniform_expansion(a, A, n)
    res = trap(fun, n, tmp)
    tmp_1 = uniform_expansion_1(a, A, n)
    res_1 = trap(fun, n, tmp_1)
    print('Звичайне розбиття ', res)
    print('Змінене розбиття: ', res_1)


def fun_2(eps, a, n):
    fun = lambda x: np.sin(x) / (x ** 2 + 1)
    A = np.tan((np.pi / 2) - eps)
    tmp = uniform_expansion(a, A, n)
    res = trap(fun, n, tmp)
    tmp_1 = uniform_expansion_1(a, A, n)
    res_1 = trap(fun, n, tmp_1)
    print('Звичайне розбиття ', res)
    print('Змінене розбиття: ', res_1)


def fun_3(eps, a, n):
    fun = lambda x: np.sin(x) / np.sqrt(1-x)
    A = 1 - eps ** 2 / 4
    tmp = uniform_expansion(a, A, n)
    res = trap(fun, n, tmp)
    tmp_1 = uniform_expansion_1(a, A, n)
    res_1 = trap(fun, n, tmp_1)
    print('Звичайне розбиття ', res)
    print('Змінене розбиття: ', res_1)


def fun_4(eps, a, n):
    fun = lambda x: (np.cos(x) - 1) / x ** 2
    A = eps ** (-0.25)
    tmp = uniform_expansion(a, A, n)
    res = trap(fun, n, tmp)
    tmp_1 = uniform_expansion_1(a, A, n)
    res_1 = trap(fun, n, tmp_1)
    print('Звичайне розбиття ', -res)
    print('Змінене розбиття: ', -res_1)


if __name__ == '__main__':
    print('Task 1')
    res1 = scipy.integrate.quad(lambda x: np.exp(-x ** 2), 0, np.inf)[0]
    print('Точний резултат - ', res1)
    print('n = 1000')
    fun_1(10**-9, 0, 1000)
    print('---------------')
    print('n = 10000')
    fun_1(10**-9, 0, 10000)

    print('\n')

    print('Task 2')
    res2 = scipy.integrate.quad(lambda x: np.sin(x) / (x ** 2 + 1), 0, np.inf)[0]
    print('Точний резултат - ', res2)
    print('n = 1000')
    fun_2(10 ** -2, 0, 1000)
    print('---------------')
    print('n = 10000')
    fun_2(10 ** -2, 0, 10000)

    print('\n')

    print('Task 3')
    res3 = scipy.integrate.quad(lambda x: np.sin(x) / np.sqrt(1 - x), 0, 1)[0]
    print('Точний резултат - ', res3)
    print('n = 1000')
    fun_3(10 ** -9, 0, 1000)
    print('---------------')
    print('n = 10000')
    fun_3(10 ** -9, 0, 10000)

    print('\n')

    print('Task 4')
    res4 = scipy.integrate.quad(lambda x: np.sin(x) / x, 0, 10**(-9*(-0.25)))[0]
    print('Точний резултат - ', res4)
    print('n = 1000')
    fun_4(10 ** -9, 0, 1000)
    print('---------------')
    print('n = 10000')
    fun_4(10 ** -9, 0, 10000)
