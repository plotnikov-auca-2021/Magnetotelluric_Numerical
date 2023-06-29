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
    # sigma_k = x[0], sigma_j = x[1], sum_of_layer_thickness = x[2]
    f1 = x[0] * np.tanh(np.sqrt(-1j * w[0] * mu0) * np.sqrt(x[0]) * x[2]) / x[1]
    f2 = np.real(x[0] * impedance[0] / np.sqrt(-1j * w[0] * mu0))
    f3 = x[0] * x[0] * np.real(impedance[2] * np.tanh(np.sqrt(-1j * w[2] * mu0) * np.sqrt(x[0]) * x[2])) \
         / (x[1] * np.sqrt(-1j * w[2] * mu0))
    return np.array([f1, f2, f3], dtype=complex)


def g_real(x):
    return np.real(g(x))


def g_imag(x):
    return np.imag(g(x))


def inverse_problem(x_initial, tol=1e-6):
    res = root(g_real, x_initial, method='hybr', tol=tol)
    if res.success:
        return res.x
    else:
        return None


x_initial = np.array([100, 100, 100], dtype=float)
roots = inverse_problem(x_initial)

if roots is not None:
    print(f"Roots: {roots}")
else:
    print("Roots cannot be found.")
