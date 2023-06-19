import cmath
import numpy as np
import matplotlib.pyplot as plt
from cmath import tanh, inf


def newton(f, Df, x0, epsilon, max_iter):
    '''
    Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.
    '''
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after', n, 'iterations.')
            print("Error: ", abs(fxn))
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn / Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None


def geometric_progression(a, r, n):
    progression = [a * r ** i for i in range(n)]
    return progression


a = 0.01  # First term
r = 2  # Common ratio
n = 3  # Number of terms

progression = geometric_progression(a, r, n)
w = np.array(progression)  # Frequencies (Hz)
S = len(w)  # number of frequencies

impedance = [0.0025036617773017845 - 0.002385335925357967j, 0.0035366887552882986 - 0.0033048932764242604j,
             0.004990678181465487 - 0.00454046292033315j]
for s in range(S):
    def phi(sigma_j, sigma_k, sum_of_layer_thickness):
        return (1 + (sigma_j / sigma_k) * tanh(np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness)) \
            - (1 + (sigma_k / sigma_j) * tanh(np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness)) \
            * (sigma_k / np.sqrt(-1j * w[s] * mu0)) * impedance[s]


    def p(x):
        return phi(x, x, x)


    def Dp(x):
        sigma_j, sigma_k, sum_of_layer_thickness = vars
        jacobi_matrix = np.zeros((S, 3))
        for s in range(S):
            jacobi_matrix[s, 0] = (np.tanh(np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness)
                                   - sigma_k * np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness
                                   * (1 / np.cosh(
                        np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness)) ** 2) \
                                  / sigma_k
            jacobi_matrix[s, 1] = -np.tanh(np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness) \
                                  / (sigma_j * sigma_k)
            jacobi_matrix[s, 2] = (sigma_j / sigma_k) * np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) \
                                  * (1 / np.cosh(
                np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness)) ** 2 \
                                  * (sigma_j + sigma_k * np.tanh(
                np.sqrt(-1j * w[s] * mu0) * np.sqrt(sigma_k) * sum_of_layer_thickness))
        return jacobi_matrix


    # Set the initial guess, tolerance, and maximum number of iterations
    x0 = 1
    epsilon = 1e-10
    max_iter = 10
    mu0 = 4 * np.pi * 1e-7

    # Use the Newton's method to find the solution
    approx = newton(p, Dp, x0, epsilon, max_iter)
    print("Value for frequency", s + 1, ":", approx)
