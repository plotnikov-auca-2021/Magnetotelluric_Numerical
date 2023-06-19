import numpy as np
import pandas as pd


def newton_method(f, f_prime, x0, tolerance, max_iterations):
    """
    Newton's iterative method to find the root of a function.

    Parameters:
    - f: The function for which we want to find the root.
    - f_prime: The derivative of the function f.
    - x0: The initial guess for the root.
    - tolerance: The desired level of accuracy.
    - max_iterations: The maximum number of iterations allowed.

    Returns:
    - A tuple containing the root found and the number of iterations performed.
    """
    x = x0
    iterations = 0

    while abs(f(x)) > tolerance and iterations < max_iterations:
        x = x - f(x) / f_prime(x)
        iterations += 1

    return x, iterations


# Define the function and its derivative
def f(x):
    return x ** 3 - 2 * x - 5


def f_prime(x):
    return 3 * x ** 2 - 2


# Define the initial guess, tolerance, and maximum number of iterations
x0 = 2.0
tolerance = 1e-6
max_iterations = 100

# Perform Newton's method
root, iterations = newton_method(f, f_prime, x0, tolerance, max_iterations)

# Create a table to display the results
data = {'Iteration': range(1, iterations + 1),
        'Approximated Root': [newton_method(f, f_prime, x0, tolerance, i)[0] for i in range(1, iterations + 1)],
        'Absolute Error': [abs(f(newton_method(f, f_prime, x0, tolerance, i)[0])) for i in range(1, iterations + 1)]}

df = pd.DataFrame(data)
print(df)
