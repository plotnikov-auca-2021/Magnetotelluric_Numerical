import numpy as np


def secant_method(f, x0, x1, tol=1e-10, max_iter=1000):
    """
    Implements the secant method for solving a system of equations.

    Args:
        f (callable): Function that returns a numpy array of equations given an array of variables.
        x0 (numpy.ndarray): Initial guess for the solution.
        x1 (numpy.ndarray): Second initial guess for the solution.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        numpy.ndarray: Solution of the system of equations.
    """
    xk = x0
    xk_minus1 = x1
    fk = f(xk)
    fk_minus1 = f(xk_minus1)

    for _ in range(max_iter):
        delta_xk = xk - xk_minus1
        delta_fk = fk - fk_minus1

        if np.all(np.abs(delta_xk) < tol):
            return xk

        Jk_minus1 = np.outer(delta_fk, delta_xk) / np.linalg.norm(delta_xk) ** 2
        Jk_minus1_pseudo_inv = np.linalg.lstsq(Jk_minus1, np.eye(len(x0)), rcond=None)[0]

        xk_plus1 = xk - Jk_minus1_pseudo_inv @ fk
        fk_plus1 = f(xk_plus1)

        xk, xk_minus1 = xk_plus1, xk
        fk, fk_minus1 = fk_plus1, fk

    raise RuntimeError("Secant method did not converge within the maximum number of iterations.")


# Example usage:

# Define the system of equations
def equations(x):
    return np.array([
        x[0] ** 2 - x[1] - 1,
        x[0] - x[1] ** 2 + 1
    ])


# Initial guesses
x0 = np.array([0.0, 0.0])
x1 = np.array([2.0, 5.0])

# Solve the system of equations using the secant method
solution = secant_method(equations, x0, x1)

print("Solution:", solution)
