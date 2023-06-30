import numpy as np

mu0 = 4 * np.pi * 1e-7
w = [0.01, 0.02, 0.04]
impedance = [
    0.0025036617773017845 - 0.002385335925357967j,
    0.0035366887552882986 - 0.0033048932764242604j,
    0.004990678181465487 - 0.00454046292033315j,
]


def g(x):
    scaling_factor = 1e9
    scaled_x = x / scaling_factor

    f1 = scaled_x[0] * np.tanh(np.sqrt(-1j * w[0] * mu0) * np.sqrt(scaled_x[0]) * scaled_x[2]) / scaled_x[1]
    f2 = scaled_x[0] * impedance[0] / np.sqrt(-1j * w[0] * mu0)
    f3 = scaled_x[0] * scaled_x[0] * impedance[2] * np.tanh(
        np.sqrt(-1j * w[2] * mu0) * np.sqrt(scaled_x[0]) * scaled_x[2]) \
         / (scaled_x[1] * np.sqrt(-1j * w[2] * mu0))

    return np.array([f1, f2, f3])


def d(x):
    scaling_factor = 1e9
    scaled_x = x / scaling_factor
    sqrt_term = np.sqrt(-1j * w[0] * mu0) * np.sqrt(scaled_x[0]) * scaled_x[2]

    d1 = (1 / scaled_x[1]) * np.tanh(sqrt_term) - (1 + (scaled_x[1] / scaled_x[0]) * np.tanh(sqrt_term)) * (
            scaled_x[0] / np.sqrt(-1j * w[0] * mu0)) * impedance[0]

    d2 = -(scaled_x[0] / (scaled_x[1] ** 2)) * np.tanh(sqrt_term) + (scaled_x[0] / scaled_x[1]) * np.tanh(sqrt_term) + (
            1 + (scaled_x[1] / scaled_x[0]) * np.tanh(sqrt_term)) * (1 / np.sqrt(-1j * w[1] * mu0)) * impedance[1]

    d3 = (scaled_x[0] / scaled_x[1]) * np.sqrt(-1j * w[2] * mu0) * np.sqrt(scaled_x[0]) * (np.cosh(sqrt_term)) ** (
        -2) + (1 + (scaled_x[1] / scaled_x[0]) * np.tanh(sqrt_term)) * (
                 scaled_x[0] / np.sqrt(-1j * w[2] * mu0)) * impedance[2] * np.sqrt(-1j * w[2] * mu0) * np.sqrt(
        scaled_x[0]) * (np.cosh(sqrt_term)) ** (-2)
    return np.array([[d1, d1, d3], [d1, d1, d3], [d2, d2, d3]])


def h(x):
    scaling_factor = 1e9
    scaled_x = x / scaling_factor
    sqrt_term1 = np.sqrt(-1j * w[0] * mu0) * np.sqrt(scaled_x[0]) * scaled_x[2]
    sqrt_term2 = np.sqrt(-1j * w[2] * mu0) * np.sqrt(scaled_x[0]) * scaled_x[2]
    cosh1 = np.cosh(sqrt_term1)
    cosh2 = np.cosh(sqrt_term2)

    h11 = (1 / scaled_x[1]) * np.sqrt(-1j * w[0] * mu0) * impedance[0] * (np.cosh(sqrt_term1)) ** (-2)
    h12 = (-scaled_x[0] / (scaled_x[1] ** 2)) * np.sqrt(-1j * w[0] * mu0) * impedance[0] * (
        np.cosh(sqrt_term1)) ** (-2) + (scaled_x[0] / scaled_x[1]) * np.sqrt(-1j * w[0] * mu0) * impedance[0] * (
              np.cosh(sqrt_term1)) ** (-2)
    h13 = (1 + (scaled_x[1] / scaled_x[0]) * np.tanh(sqrt_term1)) * (
            scaled_x[0] / np.sqrt(-1j * w[0] * mu0)) * impedance[0] * (
                  np.sqrt(-1j * w[0] * mu0) * np.sqrt(scaled_x[0]) * (np.sinh(sqrt_term1) / cosh1 ** 2) + (
                  1 / np.sqrt(-1j * w[0] * mu0)) * impedance[0] * (
                          -2 * np.sinh(sqrt_term1) / cosh1 ** 3 * sqrt_term1))
    h22 = 2 * (scaled_x[0] / (scaled_x[1] ** 3)) * np.sqrt(-1j * w[1] * mu0) * impedance[1] * (
        np.cosh(sqrt_term2)) ** (-2) - (scaled_x[0] / (scaled_x[1] ** 2)) * np.sqrt(-1j * w[1] * mu0) * impedance[1] * (
              np.cosh(sqrt_term2)) ** (-2) - (scaled_x[0] / scaled_x[1]) * np.sqrt(-1j * w[1] * mu0) * impedance[1] * (
                  2 * np.sinh(sqrt_term2) / cosh2 ** 3 * sqrt_term2)
    h23 = (1 + (scaled_x[1] / scaled_x[0]) * np.tanh(sqrt_term2)) * (
            scaled_x[0] / np.sqrt(-1j * w[2] * mu0)) * impedance[2] * (
                  np.sqrt(-1j * w[2] * mu0) * np.sqrt(scaled_x[0]) * (np.sinh(sqrt_term2) / cosh2 ** 2) + (
                  1 / np.sqrt(-1j * w[2] * mu0)) * impedance[2] * (
                          -2 * np.sinh(sqrt_term2) / cosh2 ** 3 * sqrt_term2))
    h33 = 2 * (scaled_x[0] / (scaled_x[1] ** 2)) * np.sqrt(-1j * w[2] * mu0) * impedance[2] * np.sqrt(scaled_x[0]) * (
        np.cosh(sqrt_term2)) ** (-2) - (scaled_x[0] / scaled_x[1]) * np.sqrt(-1j * w[2] * mu0) * impedance[2] * (
                  2 * np.sinh(sqrt_term2) / cosh2 ** 3 * sqrt_term2)

    return np.array([[h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])


def inverse_halley(guess, epsilon=1e-6, max_iterations=1000):
    x = guess.copy()
    for i in range(max_iterations):
        f_val = g(x)
        J_val = d(x)
        H_val = h(x)
        step = np.linalg.solve(J_val + 0.5 * H_val @ J_val, -f_val)
        x += np.real(step)
        if np.linalg.norm(step) < epsilon:
            return x
    raise ValueError("Halley's method did not converge.")


# Example usage
initial_guess = np.array([1, 1, 100], dtype=np.float64)
solution = inverse_halley(initial_guess)
print("Solution:", solution)
