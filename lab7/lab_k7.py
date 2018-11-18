import time, numpy as np


def brutforce(A, B):
    ans = []
    for i in range(len(A)):
        ans.append(str(float(sum([A[k] * B[i - k] for k in range(i + 1)]))))
    return ans


def fourier_tr(f):
    g = []
    N = len(f)

    const = np.exp(-2 * 1j * np.pi / N)
    c = [const ** k for k in range(N)]
    g.append([f[k]for k in range(0, N)])
    for i in range(1, N):
        g.append([g[-1][k] * c[k] for k in range(0, N)])
    return [sum(k) for k in g]


def reverse_fourier_tr(g):
    f = []
    N = len(g)
    const = np.exp(2 * 1j * np.pi / N)
    c = [const ** k for k in range(N)]
    f.append([g[k] for k in range(0, N)])
    for i in range(1, N):
        f.append([f[-1][k] * c[k] for k in range(0, N)])
    return [sum(k) / N for k in f]


def task1():
    A = [i for i in range(1, 101)]
    B = [i for i in range(100, 0, -1)]
    N = len(A)
    M = len(B)
    A += [0] * (M - 1)
    B += [0] * (N - 1)
    ans1 = brutforce(A, B)

    #start_time = time.time()
    G1 = fourier_tr(A)
    G2 = fourier_tr(B)
    G = [G1[i] * G2[i] for i in range(len(G1))]
    ans2 = list(str(round(z.real)) for z in reverse_fourier_tr(G))
    #print(time.time() - start_time)

    #start_time = time.time()
    G_1 = np.fft.fft(A)
    G_2 = np.fft.fft(B)
    G_ = [G_1[i] * G_2[i] for i in range(len(G_1))]
    ans3 = list(str(round(z.real)) for z in np.fft.ifft(G_))
    #print(time.time() - start_time)
    return ans1, ans2, ans3


def task2():
    def rez(f):
        return str(sum(f[i] * 10 ** i for i in range(len(f))))

    A = list(''.join(map(str, [i for i in range(1, 101)])))
    B = list(''.join(map(str, [i for i in range(100, 0, -1)])))
    N = len(A)
    G1 = fourier_tr(list(int(r) for r in A) + [0] * (N - 1))
    G2 = fourier_tr(list(int(r) for r in B) + [0] * (N - 1))
    G = [G1[i] * G2[i] for i in range(len(G1))]
    ans2 = list(int(round(z.real)) for z in reverse_fourier_tr(G))

    G_1 = np.fft.fft(A + [0] * (N - 1))
    G_2 = np.fft.fft(B + [0] * (N - 1))
    G_ = [G_1[i] * G_2[i] for i in range(len(G_1))]
    ans3 = list(int(round(z.real)) for z in np.fft.ifft(G_))
    return rez(ans2), rez(ans3)


def task3(j):
        A = [i for i in range(1, j + 1)]
        B = [i for i in range(j, 0, -1)]
        N = len(A)

        A += [0] * (N - 1)
        B += [0] * (N - 1)
        start_time = time.time()
        G1 = fourier_tr(A)
        G2 = fourier_tr(B)
        G = [G1[i] * G2[i] for i in range(len(G1))]
        ans1 = list(int(round(z.real)) for z in reverse_fourier_tr(G))
        time1 = time.time() - start_time

        start_time = time.time()
        G_1 = np.fft.fft(A)
        G_2 = np.fft.fft(B)
        G_ = [G_1[i] * G_2[i] for i in range(len(G_1))]
        ans2 = list(int(round(z.real)) for z in np.fft.ifft(G_))
        time2 = time.time() - start_time
        return str(time1),  str(time2)


if __name__ == '__main__':
    # print(fourier_tr(np.arange(1, 101)))
    with open('lab7.txt', 'w') as f:
        f.write('Завдання1\n'
                'f(x) = 1 + 2x + 3x^2 + ... + 100x^99\n'
                'g(x) = 100 + 99x + 98x^2 + ... + x^99\n'
                'f(x) * g(x) = \n')
        t1 = task1()
        f.write('Явно:                    ' + ' '.join(t1[0])+'\n')
        f.write('Дискретне перетворення:  ' + ' '.join(t1[1]) + '\n')
        f.write('Швидке перетворення:     ' + ' '.join(t1[2]) + '\n\n\n')

        f.write('Завдання2\n'
                'a = 1234...9899100\n'
                'b = 1009998...4321\n'
                'a * b = \n')
        t2 = task2()
        f.write('Дискретне перетворення:  ' + t2[0] + '\n')
        f.write('Швидке перетворення:     ' + t2[1] + '\n\n\n')

        f.write('Завдання3 a)\n'
                'f(x) = 1 + 2x + 3x^2 + ... + 1000x^999\n'
                'g(x) = 1000 + 99x + 98x^2 + ... + x^999\n'
                'f(x) * g(x)\n')
        t3 = task3(1000)
        f.write('Дискретне перетворення(час виконання):  ' + t3[0] + '\n')
        f.write('Швидке перетворення(час виконання):     ' + t3[1] + '\n\n')

        # f.write('Завдання3 b)\n'
        #         'f(x) = 1 + 2x + 3x^2 + ... + 10000x^9999\n'
        #         'g(x) = 10000 + 99x + 98x^2 + ... + x^9999\n'
        #         'f(x) * g(x)\n')
        # t3 = task3(10000)
        # f.write('Дискретне перетворення(час виконання):  ' + t3[0] + '\n')
        # f.write('Швидке перетворення(час виконання):     ' + t3[1])