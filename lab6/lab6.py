"""
by Roman Polishchenko
2 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""
import numpy as np


def integrate(func, r, n=1000, split=None):
    """
    Інтеграл func(x) dx на інтервалі (0, +00)
    :param func: функція
    :param r: А - права границя
    :param n: кіс-ть інтервалів
    :param split: left/right - нелінійне розбиття, None/False – лінійне
    :return: значення інтеграла
    """
    a = 0
    b = r
    res = 0
    if not split:
        for i in range(1, n):
            x1 = a + (b - a) * i / n
            x2 = a + (b - a) * (i + 1) / n
            res += (func(x1) + func(x2)) / 2 * (x2 - x1)

    elif split == 'left':
        for i in range(1, n):
            x1 = a + ((b - a) * i**2)/n**2
            x2 = a + ((b - a) * (i + 1)**2)/n**2
            res += (func(x1) + func(x2)) / 2 * (x2 - x1)

    elif split == 'right':
        for i in range(1, n):
            x1 = a + ((b - a) * (n - i)**2)/n**2
            x2 = a + ((b - a) * (n - i + 1)**2)/n**2
            res += (func(x1) + func(x2)) / 2 * (x2 - x1)
    else:
        print('Wrong input!')
    return res


if __name__ == '__main__':
    eps = 10**-9
    n1 = 1000
    n2 = 10000
    with open('output.txt', 'w') as file:
        print('Завдання 1 a)\nƒ(x)dx = exp(-x^2), a = 0, b = +inf', file=file)
        A = np.sqrt(np.log(1/(2*eps)))
        print('A =', A, file=file)
        print('\tn =', n1, file=file)
        print('Лінійне розбиття:', integrate(lambda x: np.exp(-(x**2)), A, n1), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: np.exp(-x**2), A, n1, 'right'), file=file)
        print('\tn =', n2, file=file)
        print('Лінійне розбиття:', integrate(lambda x: np.exp(-(x ** 2)), A, n2), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: np.exp(-x ** 2), A, n2, 'right'), file=file)

        print('\nЗавдання 1 b)\nƒ(x)dx = sin(x)/(x**2 + 1), a = 0, b = +inf\n', file=file)
        # A = np.pi / 2 - eps
        A = 2 * np.pi / 3 - eps
        print('A =', A, file=file)
        print('\tn =', n1, file=file)
        print('Лінійне розбиття:', integrate(lambda x: np.sin(x) / (x**2 + 1), A, n1), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: np.sin(x) / (x**2 + 1), A, n1, 'right'), file=file)
        print('\tn =', n2, file=file)
        print('Лінійне розбиття:', integrate(lambda x: np.sin(x) / (x**2 + 1), A, n2), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: np.sin(x) / (x**2 + 1), A, n2, 'right'), file=file)

        print('\nЗавдання 1 c)\nƒ(x)dx = sin(x)/sqrt(1-x), a = 0, b = 1\n', file=file)
        # A = 1 - eps**2/4
        # через проблеми з точністю обрахунків, пітон не хоче рахувати
        # так і видає помилку, тому вводимо значення вручну
        A = 0.998  # вже при 0.999 видає помилку [бо вважає, що в знаменнику нуль]
        print('A =', A, file=file)
        print('\tn =', n1, file=file)
        print('Лінійне розбиття:', integrate(lambda x: np.sin(x) / np.sqrt(1-x), A, n1), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: np.sin(x) / np.sqrt(1-x), A, n1, 'right'), file=file)
        print('\tn =', n2, file=file)
        print('Лінійне розбиття:', integrate(lambda x: np.sin(x) / np.sqrt(1-x), A, n2), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: np.sin(x) / np.sqrt(1-x), A, n2, 'right'), file=file)

        print('\nЗавдання 2\nƒ(x)dx = sin(x)/x, a = 0, b = +inf\n', file=file)
        print('Після інтегрування частинами отримали функцію (-cos(x) + 1) / x**2\n', file=file)
        A = 100
        print('A =', A, file=file)
        print('\tn =', n1, file=file)
        print('Лінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x**2, A, n1), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x**2, A, n1, 'right'), file=file)
        print('\tn =', n2, file=file)
        print('Лінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x**2, A, n2), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x**2, A, n2, 'right'), file=file)

        A = 1000
        print('\nA =', A, file=file)
        print('\tn =', n1, file=file)
        print('Лінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x ** 2, A, n1), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x ** 2, A, n1, 'right'), file=file)
        print('\tn =', n2, file=file)
        print('Лінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x ** 2, A, n2), file=file)
        print('Нелінійне розбиття:', integrate(lambda x: (-np.cos(x) + 1) / x ** 2, A, n2, 'right'), file=file)

        print('\nПомічаємо, що при збільшенні правої межі точність інтегрування по лінійному\n'
              'розбиттю зменшується, а по нелінійному – збільшується.', file=file)
