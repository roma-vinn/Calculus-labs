from math import cos


def f1(x):
    # [0, 1]
    return x**10 - 0.1*x - 0.01


def f2(x):
    # [-0.4. 0.6]
    # [-1.5, 1.6]
    from math import sin
    return 6*sin(x**7) + x**21 - 6*x**14


def my_func(x):
    # [0, 2]
    return cos(x)


def solve_half(func, a, b, eps=10**-10):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param eps: precision measure
    :return: c, where f(c) = 0
    """
    left = a
    right = b
    precision_1 = abs(left - right)
    center = (left + right) / 2
    precision_2 = abs(func(center))
    while precision_1 > eps and precision_2 > eps:
        if func(left)*func(center) < 0:
            right = center
        else:
            left = center
        if func(center) == 0:
            return center
        precision_1 = abs(left - right)
        precision_2 = abs(func(center))
        center = (left + right) / 2
    return center


def solve_tangent(func, a, b, eps=10**-10):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param eps: precision measure
    :return: c, where f(c) = 0
    """
    x0 = a
    x1 = b
    xn = x1 - func(x0)*(x1 - x0)/(func(x1) - func(x0))
    precision = abs(func(xn))
    while precision > eps:
        x0, x1, xn = x1, xn, x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        precision = abs(func(xn))
    return xn


def solve_secant(func, a, b, eps=10**-10):
    """
    :param func: f(x)
    :param a: lower bound
    :param b: upper bound
    :param eps: precision measure
    :return: c, where f(c) = 0
    """

    def dif(_func, x_0, x_1):
        """
        :param _func: f(x)
        :param x_0: float
        :param x_1: float
        :return: F'(x0)
        """
        res = (_func(x_1) - _func(x_0))/(x_1 - x_0)
        return res
    
    x0 = a
    x1 = b
    precision = abs(func(x1))
    while precision > eps:
        x0, x1 = x1, x1 - func(x1) / dif(func, x0, x1)
        precision = abs(func(x1))
    return x1


with open('output_solving.txt', 'w') as file:
    print('f(x) = x**10 - 0.1*x - 0.01 на проміжку [0, 1]', file=file)
    c = solve_half(f1, 0, 1)
    print('\tМетод половинного поділу: c = {}, f(c) = {}'.format(c, f1(c)), file=file)
    c = solve_secant(f1, 0, 1)
    print('\tМетод січних: c = {}, f(c) = {}'.format(c, f1(c)), file=file)
    c = solve_tangent(f1, 0, 1)
    print('\tМетод дотичних: c = {}, f(c) = {}'.format(c, f1(c)), file=file)

    print('f(x) = 6*sin(x**7) + x**21 - 6*x**14 на проміжку [-0.4, 0.6]', file=file)
    c = solve_half(f2, -0.4, 0.6)
    print('\tМетод половинного поділу: c = {}, f(c) = {}'.format(c, f2(c)), file=file)
    c = solve_secant(f2, -0.4, 0.6)
    print('\tМетод січних: c = {}, f(c) = {}'.format(c, f2(c)), file=file)
    c = solve_tangent(f2, -0.4, 0.6)
    print('\tМетод дотичних: c = {}, f(c) = {}'.format(c, f2(c)), file=file)

    print('f(x) = 6*sin(x**7) + x**21 - 6*x**14 на проміжку [-1.4, 1.6]', file=file)
    c = solve_half(f2, -1.4, 0.6)
    print('\tМетод половинного поділу: c = {}, f(c) = {}'.format(c, f2(c)), file=file)
    c = solve_secant(f2, -1.4, 0.6)
    print('\tМетод січних: c = {}, f(c) = {}'.format(c, f2(c)), file=file)
    c = solve_tangent(f2, -1.4, 0.6)
    print('\tМетод дотичних: c = {}, f(c) = {}'.format(c, f2(c)), file=file)

    print('my_func(x) = cos(x) на проміжку [0, 2]', file=file)
    c = solve_half(my_func, 0, 2)
    print('\tМетод половинного поділу: c = {}, f(c) = {}'.format(c, my_func(c)), file=file)
    c = solve_secant(my_func, 0, 2)
    print('\tМетод січних: c = {}, f(c) = {}'.format(c, my_func(c)), file=file)
    c = solve_tangent(my_func, 0, 2)
    print('\tМетод дотичних: c = {}, f(c) = {}'.format(c, my_func(c)), file=file)
