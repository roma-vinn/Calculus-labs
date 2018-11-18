import matplotlib.pyplot as plt


def g(t):
    return 3*t**2 - 2*t**3


def func_approx(t0, tn, k0, y0, n, g_func):
    delta = (tn-t0)/n
    _X = [t0]
    _Y = [y0]
    for i in range(1, n+1):
        _x = t0 + delta*i
        _y = (k0*_Y[i-1] + g_func(_X[i-1]))*delta + _Y[i-1]
        _X.append(_x)
        _Y.append(_y)
    return _X, _Y


x, y = func_approx(0, 1, 2, 0, 100, g)
plt.plot(x, y, 'ro')
plt.show()
