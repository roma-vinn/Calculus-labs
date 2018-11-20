"""by Roman Polishchenko2 course, comp mathTaras Shevchenko National University of Kyivemail: roma.vinn@gmail.com"""from copy import deepcopyfrom time import clockfrom random import uniformfrom itertools import accumulatefrom math import sin, expdef add_step(i: int, left: list, a: list, step: float, right: list) -> list:    """ Рекурсивний метод, що додає до вектора покоординатно,    починаючи з останньої координати, величину рівну step    :param i: поточна координата    :param left: список координат "лівої нижньої точки" умовного гіперкуба    :param a: список координат поточної точки    :param step: крок    :param right: список координат "правої верхньої точки" умовного гіперкуба    :return: [], якщо було виконане останнє додавання;             інакше – список координат поточної точки    """    a[i] += step    if a[i] > right[i]:        if i != 0:            a[i] = left[i]            return add_step(i-1, left, a, step, right)        else:            return []    else:        return adef get_dots(dot: list, step: float) -> list:    """ Отримати координати всіх кутів та цента гіперкуба, якщо відомий лівий нижній кут    :param dot: координати лівого нижнього кута    :param step: степінь розбиття (крок)    :return: список координат потрібних точок    """    m = len(dot)    dots = [list(map(lambda x: x + step / 2, dot))]    for i in range(1, 2**m):        tmp = deepcopy(dot)        num = bin(i)[2:].zfill(m)        for k in range(m):            if num[k] == '1':                tmp[k] += step        dots.append(tmp)    return dotsdef measure(array: callable, borders: tuple, n=4) -> tuple:    """ Функція для обчислення міри Жордана множини заданої функцією func, обмеженою        гіперкубом, що заданий координатами двох діаметрально протилежних точок,        записаних в borders, зі степенем розбиття n.    :param array: функція, що задає множину; повертає True, якщо точка належить множині, False – інакше    :param borders: кортеж, що містить координати двох крайніх точок    :param n: степінь розбиття    :return: внутрішня та зовнішня міри Жордана, помилка    """    left, right = borders    step = 1/2**n    curr = deepcopy(left)    m = step ** len(curr)    # A^n    upper = 0    # A_n    lower = 0    while True:        curr = add_step(len(curr)-1, left, curr, step, right)        if not curr:            break        else:            flag_u = False            flag_l = True            for dot in get_dots(curr, step):                if array(dot):                    flag_u = True                else:                    flag_l = False            if flag_u:                upper += m            if flag_l:                lower += m    error = upper - lower    return lower, upper, errordef measure_monte_carlo(array: callable, borders: tuple, n=100) -> float:    """ Функція для обчислення міри Жордана методом Монте-Карло множини заданої        функцією func, обмеженою гіперкубом, що заданий координатами двох        діаметрально протилежних точок, записаних в borders, де n - к-сть обраних точок.    :param array: функція, що задає множину; повертає True, якщо точка належить множині, False – інакше    :param borders: кортеж, що містить координати двох крайніх точок    :param n: к-сть обраних точок    :return: міра Жордана    """    inside = 0    borders = list(zip(borders[0], borders[1]))    _measure = map(lambda x: abs(x[0] - x[1]), borders)    _measure = list(accumulate(_measure, lambda x, y: x * y))[-1]    for i in range(n):        dot = list(map(lambda x: uniform(x[0], x[1]), borders))        if array(dot):            inside += 1    _measure *= inside/n    return _measuredef integral(func: callable, array: callable, borders, n=4) -> tuple:    """ Функція для обчислення інтеграла функції func по множині array,        обмеженій гіперкубом з крайніми точками borders, де n - степінь розбиття    :param func: підінтегральна функція    :param array: функція, що задає множину    :param borders: границі гіперкуба, що обмежує множину    :param n: степінь розбиття    :return: інтеграл по внутрішній та зовнішній мірі Жордана, помилка    """    left, right = borders    step = 1/2**n    curr = deepcopy(left)    m = step ** len(curr)    upper_m = 0    upper_f = 0    lower_m = 0    lower_f = 0    sup = abs(func(borders[0]))    while True:        curr = add_step(len(curr)-1, left, curr, step, right)        if not curr:            break        else:            flag_u = False            flag_l = True            for dot in get_dots(curr, step):                if array(dot):                    flag_u = True                else:                    flag_l = False            f_psi = func(list(map(lambda x: x + step / 2, curr)))            sup = max(abs(f_psi), sup)            if flag_u:                upper_m += m                upper_f += f_psi            if flag_l:                lower_m += m                lower_f += f_psi    # lower = lower_f * lower_m    # upper = upper_f * upper_m    lower = lower_f * m    upper = upper_f * m    error = (upper_m - lower_m) * sup    return lower, upper, errordef integral_monte_carlo(func: callable, array: callable, borders, n=100) -> float:    """ Функція для обчислення інтеграла методом Монте-Карло функції func по множині array,        обмеженій гіперкубом з крайніми точками borders, де n - к-сть обраних точок    :param func: підінтегральна функція    :param array: функція, що задає множину; повертає True, якщо точка належить множині, False – інакше    :param borders: кортеж, що містить координати двох крайніх точок    :param n: к-сть обраних точок    :return: інтеграл    """    inside = 0    borders = list(zip(borders[0], borders[1]))    _measure = map(lambda x: abs(x[0] - x[1]), borders)    _measure = list(accumulate(_measure, lambda x, y: x * y))[-1]    f_psi = 0    for i in range(n):        dot = list(map(lambda x: uniform(x[0], x[1]), borders))        if array(dot):            f_psi += func(dot)            inside += 1    _measure *= inside / n    _integral = f_psi/inside * _measure    return _integraldef task1a():    """Task 1 a)\nThe set is limited by x1^4 + x2^4 <= 1    """    def array(vec):        """ x1^2 + x2^2 <= 1        """        return vec[0] ** 4 + vec[1] ** 4 <= 1    borders = ([-1, -1], [1, 1])    res = ""    res += 'Split method:\n\n'    for N in [4, 6, 8]:        begin = clock()        lower, _, error = measure(array, borders, N)        res += '\t n = {}\n'.format(N)        res += 'Calculated measure: {}\n'.format(lower)        res += 'Error: {}\n'.format(error)        res += 'Time: {}\n\n'.format(clock() - begin)    res += 'Monte-Carlo method:\n\n'    for N in [100, 1000, 10000]:        begin = clock()        mes = measure_monte_carlo(array, borders, N)        res += '\t n = {}\n'.format(N)        res += 'Calculated measure: {}\n'.format(mes)        res += 'Time: {}\n\n'.format(clock() - begin)    return resdef task1b():    """Task 1 b)\nThe set is limited by x1^2 + x2^2 <= 1 and x1 + x2 <= x3 <= 2x1 + 3x2    """    def array(vec):        """ x1^2 + x2^2 <= 1 and x1 + x2 <= x3 <= 2x1 + 3x2        """        return vec[0] ** 2 + vec[1] ** 2 <= 1 and vec[0] + vec[1] <= vec[2] <= 2 * vec[0] + 3 * vec[1]    borders = ([-1, -1, 0], [1, 1, 3.7])    res = ""    res += 'Split method:\n\n'    for N in [4, 5]:        begin = clock()        lower, _, error = measure(array, borders, N)        res += '\t n = {}\n'.format(N)        res += 'Calculated measure: {}\n'.format(lower)        res += 'Error: {}\n'.format(error)        res += 'Time: {}\n\n'.format(clock() - begin)    res += 'Monte-Carlo method:\n\n'    for N in [100, 1000, 10000]:        begin = clock()        mes = measure_monte_carlo(array, borders, N)        res += '\t n = {}\n'.format(N)        res += 'Calculated measure: {}\n'.format(mes)        res += 'Time: {}\n\n'.format(clock() - begin)    return resdef task2():    """Task 2\nIntegral of sin(eps(x-y)) dx dy, where x^4 + y^4 <= 1    """    def func(vec):        """ f(x1, x2) = sin(eps(x1-x2))        """        return sin(exp(vec[0]-vec[1]))    def array(vec):        """ x1^2 + x2^2 <= 1        """        return vec[0] ** 4 + vec[1] ** 4 <= 1    borders = ([-1, -1], [1, 1])    res = ''    res += 'Split method:\n\n'    for N in [4, 6, 8]:        begin = clock()        lower, _, error = integral(func, array, borders, N)        res += '\t n = {}\n'.format(N)        res += 'Calculated integral: {}\n'.format(lower)        res += 'Error: {}\n'.format(error)        res += 'Time: {}\n\n'.format(clock() - begin)    res += 'Monte-Carlo method:\n\n'    for N in [100, 1000, 10000]:        begin = clock()        integ = integral_monte_carlo(func, array, borders, N)        res += '\t n = {}\n'.format(N)        res += 'Calculated integral: {}\n'.format(integ)        res += 'Time: {}\n\n'.format(clock() - begin)    return resdef task3():    """Task 3\nMeasure of x1^2 + x2^2 + ... + xm^2 <= 0.25    """    def array(vec):        """x1^2 + x2^2 + ... + xm^2 <= 0.25        """        return list(accumulate(vec, lambda x, y: x**2 + y**2))[-1] <= 0.25    def borders(dimensions):        """ Обмежуючий гіперкуб зі стороною 1.        """        return [-1] * dimensions, [1] * dimensions    res = ''    # при m > 4 працює занадто довго    for m in range(1, 5):        res += '\t\tm = {}\n\n'.format(m)        res += 'Split method:\n'        n = 4        begin = clock()        res += '\tn = {}\n'.format(n)        mes = measure(array, borders(m), n)[0]        res += 'Calculated measure: {}\n'.format(mes)        res += 'Time: {}\n\n'.format(clock() - begin)        res += 'Monte-Carlo method:\n'        n = 10000        res += '\tn = {}\n'.format(n)        begin = clock()        mes = measure_monte_carlo(array, borders(m), n)        res += 'Calculated measure: {}\n'.format(mes)        res += 'Time: {}\n\n'.format(clock() - begin)    return resif __name__ == '__main__':    # TASKS = [task1a, task1b, task2, task3]    TASKS = [task2]    with open('output.txt', 'w') as file:        for task in TASKS:            print(task.__doc__, file=file)            print(task(), file=file)