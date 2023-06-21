import numpy as np


def g(x):
    f1 = x[0] ** 2 - x[1] - 1
    f2 = x[0] - x[1] ** 2 + 1
    return np.array([f1, f2])


def d(x):
    return np.array([[2 * x[0], -1], [1, -2 * x[1]]])


def fixed_point_iteration(x0, tol=0.001, max_iters=1000):
    i = 0
    while i < max_iters:
        x1 = x0 - np.linalg.inv(d(x0)) @ g(x0)
        if np.linalg.norm(x1 - x0) < tol:
            return x1
        x0 = x1
        i += 1
    return None


x_initial_first = np.array([100, 100])
first_root = fixed_point_iteration(x_initial_first)

if first_root is not None:
    print(f"Root: {first_root}")
else:
    print("Root cannot be found within the maximum number of iterations.")
