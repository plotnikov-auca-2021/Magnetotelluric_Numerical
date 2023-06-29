import numpy as np
from scipy.optimize import root

mu0 = 4 * np.pi * 1e-7
w = [0.01, 0.02, 0.04]
impedance = [
    0.0025036617773017845 - 0.002385335925357967j,
    0.0035366887552882986 - 0.0033048932764242604j,
    0.004990678181465487 - 0.00454046292033315j,
]


def g(x):
    f1 = x[0] * np.tanh(np.sqrt(-1j * w[0] * mu0) * np.sqrt(x[0]) * x[2]) / x[1]
    f2 = x[0] * impedance[0] / np.sqrt(-1j * w[0] * mu0)
    f3 = x[0] * x[0] * impedance[2] * np.tanh(np.sqrt(-1j * w[2] * mu0) * np.sqrt(x[0]) * x[2]) \
         / (x[1] * np.sqrt(-1j * w[2] * mu0))
    return np.array([f1, f2, f3])


def pd1(x):
    sqrt_term = np.sqrt(-1j * w[0] * mu0) * np.sqrt(x[0]) * x[2]
    return (1 / x[1]) * np.tanh(sqrt_term) - (1 + (x[1] / x[0]) * np.tanh(sqrt_term)) * (
                x[0] / np.sqrt(-1j * w[0] * mu0)) * impedance[0]


def pd2(x):
    sqrt_term = np.sqrt(-1j * w[1] * mu0) * np.sqrt(x[0]) * x[2]
    return -(x[0] / (x[1] ** 2)) * np.tanh(sqrt_term) + (x[0] / x[1]) * np.tanh(sqrt_term) + (
                1 + (x[1] / x[0]) * np.tanh(sqrt_term)) * (1 / np.sqrt(-1j * w[1] * mu0)) * impedance[1]


def pd3(x):
    sqrt_term = np.sqrt(-1j * w[2] * mu0) * np.sqrt(x[0]) * x[2]
    return (x[0] / x[1]) * np.sqrt(-1j * w[2] * mu0) * np.sqrt(x[0]) * (np.cosh(sqrt_term)) ** (-2) - (
                x[1] / x[0]) * np.sqrt(-1j * w[2] * mu0) * np.sqrt(x[0]) * (np.cosh(sqrt_term)) ** (-2) + (
                1 + (x[1] / x[0]) * np.tanh(sqrt_term)) * (x[0] / np.sqrt(-1j * w[2] * mu0)) * impedance[2] * np.sqrt(
        -1j * w[2] * mu0) * np.sqrt(x[0]) * (np.cosh(sqrt_term)) ** (-2)


def d(x):
    return np.array([[pd1(x), pd1(x), pd1(x)],
                     [pd2(x), pd2(x), pd2(x)],
                     [pd3(x), pd3(x), pd3(x)]])


def inverse_problem(x0, tol=0.001, max_iters=1000):
    for i in range(max_iters):
        x1 = x0 - np.linalg.inv(d(x0)) @ g(x0)
        if np.linalg.norm(x1 - x0) < tol:
            return x1
        x0 = x1
    return None


x_initial = np.array([100, 100, 100])
root = inverse_problem(x_initial)

if root is not None:
    print(f"Root: {root}")
else:
    print("Root cannot be found within the maximum number of iterations.")
