import numpy as np


def halley_method(f, J, H, x0, epsilon=1e-6, max_iterations=100):
    """
    Halley's method for solving systems of equations.

    Args:
        f (callable): Function that represents the system of equations.
        J (callable): Jacobian matrix function of the system of equations.
        H (callable): Hessian matrix function of the system of equations.
        x0 (numpy.ndarray): Initial guess for the solution.
        epsilon (float, optional): Error tolerance for convergence. Default is 1e-6.
        max_iterations (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        numpy.ndarray: Solution of the system of equations.
    """
    x = x0.copy()
    for i in range(max_iterations):
        f_val = f(x)
        J_val = J(x)
        H_val = H(x)
        step = np.linalg.solve(J_val + 0.5 * H_val @ J_val, -f_val)
        x += step
        if np.linalg.norm(step) < epsilon:
            return x
    raise ValueError("Halley's method did not converge.")


# Example usage:
# Define the system of equations: f(x, y) = [f1(x, y), f2(x, y)]
def f(x):
    return np.array([
        x[0] ** 2 - 4 * x[1] ** 2 - 3,
        x[0] ** 2 + x[1] ** 2 - 4
    ])


# Define the Jacobian matrix of the system of equations: J(x, y)
def J(x):
    return np.array([
        [2 * x[0], -8 * x[1]],
        [2 * x[0], 2 * x[1]]
    ])


# Define the Hessian matrix of the system of equations: H(x, y)
def H(x):
    return np.array([
        [2, 0],
        [2, 0]
    ])


# Initial guess for the solution
x0 = np.array([84.0, 1.0])

# Solve the system of equations using Halley's method
solution = halley_method(f, J, H, x0)

print("Solution:", solution)
