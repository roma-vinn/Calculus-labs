"""
by Roman Polishchenko
2 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def approx(func: callable, m: int, n: int, border: tuple, graph=False) -> tuple:
    """
    Апроксимація функції многочленом.
    :param func: функція, яку апроксимуємо
    :param m: степінь многочлена
    :param n: кількість точок розбиття
    :param border: границі значення точок
    :param graph: будувати графік, чи ні
    :return: коефіцієнти многочлена
    """
    arr_x = np.linspace(*border, n)
    matrix_a = np.array([[arr_x[j]**i for i in range(m)] for j in range(n)])
    b = func(arr_x).reshape((n, 1))
    a = np.linalg.solve(matrix_a.T.dot(matrix_a), matrix_a.T.dot(b))
    arr_y = np.zeros(n)
    for j in range(len(arr_x)):
        ans = 0
        for i in range(m):
            ans += a[i] * arr_x[j] ** i
        arr_y[j] = ans

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111)
        ax1.plot(arr_x, func(arr_x))
        ax2.scatter(arr_x, arr_y)
        plt.show()
    return arr_x, arr_y, a


def robot_placement(n=5):
    """
    Функція, що розв'язує задачу про місцезнаходження робота.
    :param n: кі-сть світильників
    :return: (x, y) - найбільш імовірні координати робота
    """
    # координати світильників
    torches_x = np.random.random((1, n)) * 4
    torches_y = np.random.random((1, n)) * 5

    # координати робота
    robot_x = np.random.random() * 4
    robot_y = np.random.random() * 5

    # виміри робота
    dx = torches_x - robot_x + np.random.randint(-1, 1) / 20  # з урахуванням помилки
    dy = torches_y - robot_y + np.random.randint(-1, 1) / 20

    # функція похибки
    def f(coord, k=n):
        x = coord[0]
        y = coord[1]
        return sum([(x - torches_x[0, i] + dx[0, i])**2 + (y - torches_y[0, i] + dy[0, i])**2
                   for i in range(k)])

    # початкова точка
    x0 = np.array([0, 0])
    res = ''
    # точні координати
    res += 'Actual robot coordinates: (%.8f, %.8f)\n' % (robot_x, robot_y)
    # мінімізуємо функцію
    supposed_x, supposed_y = scipy.optimize.minimize(f, x0).x
    # знайдені коодинати (найбільш ймовірні)
    res += 'Supposed coordinates: (%.8f, %.8f)\n' % (supposed_x, supposed_y)
    # похибка
    error = np.sqrt((supposed_x-robot_x) ** 2 + (supposed_x-robot_x)**2)
    res += 'Delta: %.8f' % error
    return res, error


def test(count=1000):
    """
    Тестуємо нашу функцію.
    :param count: кі-сть тестів
    :return: середня похибка
    """
    avg_error = 0
    for c in range(count):
        avg_error += robot_placement()[1]
    return 'Average error is %.8f\n' % (avg_error/count)


if __name__ == '__main__':
    # Завдання 1
    approx(lambda x: np.sin(x), 2, 2, (0, np.pi))
    approx(lambda x: np.sin(x), 20, 30, (0, np.pi), True)

    approx(lambda x: np.exp(x), 2, 2, (0, 1))
    approx(lambda x: np.exp(x), 20, 30, (0, 1), True)

    # Завдання 2
    with open('output.txt', 'w') as file:
        print(robot_placement()[0], file=file)
        print(test(), file=file)
