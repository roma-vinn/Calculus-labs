"""
by Roman Polishchenko
2 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""
import numpy as np
import time


def brut_force(poly_1: np.ndarray, poly_2: np.ndarray) -> np.ndarray:
    """
    Добуток поліномів зі списками коефіцієнтів poly_1 і poly_2.
    :param poly_1: коефіцієнти
    :param poly_2: коефіцієнти
    :return: коефіцієнти добутку
    """
    res_n = poly_1.size
    res = np.array([0] * res_n)
    for n in range(res_n):
        res[n] = (float(sum([poly_1[k] * poly_2[n - k] for k in range(n + 1)])))
    return res


def fourier_transform(poly: np.ndarray) -> np.ndarray:
    """
    Дискретне перетворення Фур'є.
    :param poly: коефіцієнти поліному
    :return: дпф
    """
    g = []
    n = poly.size
    const = np.exp(-2 * 1j * np.pi / n)
    c = [const ** k for k in range(n)]
    g.append([poly[k] for k in range(0, n)])
    for i in range(1, n):
        g.append([g[-1][k] * c[k] for k in range(0, n)])
    return np.array([sum(k) for k in g])


def reversed_fourier_transform(poly: np.ndarray) -> np.ndarray:
    """
    Обернене перетворення Фур'є.
    :param poly: дпф
    :return: коефіцієнти поліному
    """
    g = []
    n = poly.size
    const = np.exp(2 * 1j * np.pi / n)
    c = [const ** k for k in range(n)]
    g.append([poly[k] for k in range(0, n)])
    for i in range(1, n):
        g.append([g[-1][k] * c[k] for k in range(0, n)])
    return np.array([1/n * sum(k) for k in g])


def task1(*args, output=True):
    """
    Завдання 1.
    Перемножаємо поліноми f(x) = 1 + 2x + 3x^2 + ... + 100x^99
    та g(x) = 100 + 99x + 98x^2 + ... + x^99
    """
    # вважаємо, що першим аргументом є назва вихідного файлу
    f = args[0]
    # другим аргументом може йти номер n старший коефіцієнт полінома
    if len(args) > 1:
        n = args[1]
    else:
        n = 100
    # створюємо поліноми та дописуємо в їх коефіцієнти нулі
    a = np.arange(1, n + 1)
    b = np.arange(n, 0, -1)
    m = a.size + b.size - 1
    a.resize(m)
    b.resize(m)

    flag = time.clock()
    bf = brut_force(a, b)
    result = ''
    result += 'Явно. Час: {} '.format(time.clock() - flag)
    if output:
        result += 'Результат:\n' + str(list(map(lambda x: int(round(x)), bf)))

    if output:
        flag = time.clock()
        a_t = fourier_transform(a)
        b_t = fourier_transform(b)
        r_t = a_t * b_t
        r = reversed_fourier_transform(r_t)
        result += '\nДискретне перетворення. Час: {}'.format(time.clock() - flag)
        result += ' Результат:\n' + str(list(map(lambda x: int(round(x)), r.real)))

    flag = time.clock()
    a_t = np.fft.fft(a)
    b_t = np.fft.fft(b)
    r = np.fft.ifft(a_t*b_t)
    result += '\nШвидке перетворення. Час: {}'.format(time.clock() - flag)
    if output:
        result += ' Результат:\n' + str(list(map(lambda x: int(round(x)), r.real)))

    print(result, file=f)


def task2(f):
    """
    Завдання 2.
    Перемножити числа a та b, де
    a = 12345678910111213...100
    b = 10099989796959493...321
    """
    def result(indexes):
        return str(sum(indexes[i] * 10 ** i for i in range(len(indexes))))

    a = np.array(list(map(int, ''.join(map(str, [i for i in range(1, 101)])))))
    b = np.array(list(map(int, ''.join(map(str, [i for i in range(100, 0, -1)])))))
    m = a.size + b.size - 1
    a.resize(m)
    b.resize(m)

    flag = time.clock()
    r = reversed_fourier_transform(fourier_transform(a) * fourier_transform(b))
    r = list(map(lambda x: int(round(x)), r.real))
    print('Дискретне перетворення. Час:', time.clock() - flag, 'a * b =', file=f)
    print(result(r), file=f)

    flag = time.clock()
    r = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))
    r = list(map(lambda x: int(round(x)), r.real))
    print('\nШвидке перетворення. Час:', time.clock() - flag, 'a * b =', file=f)
    print(result(r), file=f)


def task3(f):
    """
    Завдання 3.
    Перевiрити швидкiсть перетворень Фур’є, замiнивши в задачi 1 100 на
    1000 та 10000 i замiрявши час роботи двох алгоритмiв.
    """
    for n in [1000, 10000]:
        print('n = {}'.format(n), file=f)
        task1(f, n, output=False)
        print(file=f)


if __name__ == '__main__':
    TASKS = [task1,
             task2,
             task3]
    with open('output.txt', 'w') as file:
        for t in TASKS:
            print(t.__doc__, file=file)
            t(file)
