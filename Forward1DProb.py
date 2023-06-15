import cmath
import numpy as np
import matplotlib.pyplot as plt
from cmath import tanh, phase, inf


def wave_number_k(sigma, w, mu0):
    J = len(sigma)  # number of layers
    S = len(w)  # number of frequencies

    waveNumberK = np.zeros((J, S), dtype=complex)

    for j in range(J):
        for s in range(S):
            waveNumberK[j, s] = (1 - 1j) * np.sqrt(w[s] * mu0 * sigma[j] / 2.0)

    return waveNumberK


def geometric_progression(a, r, n):
    progression = [a * r ** i for i in range(n)]
    return progression


a = 0.01  # First term
r = 2  # Common ratio
n = 27  # Number of terms

progression = geometric_progression(a, r, n)
print(progression)


def Fwd1DProblem(mu0, w, sigma, h):
    S = len(w)  # number of frequencies
    J = len(sigma)  # number of layers

    waveNumberK = wave_number_k(sigma, w, mu0)

    Z = np.zeros((J, S), dtype=complex)
    for j in range(J, 0, -1):
        for s in range(S):
            if j == J:
                Z[int(j) - 1, s] = waveNumberK[int(J) - 1, s] / sigma[int(J) - 1]
            else:
                Z[int(j) - 1, s] = (waveNumberK[int(j) - 1, s] * Z[int(j), s] - 1j * w[s] * mu0 * tanh(
                    h[int(j) - 1] * waveNumberK[int(j) - 1, s])) / \
                                   (waveNumberK[int(j) - 1, s] + sigma[int(j) - 1] * tanh(h[int(j) - 1] * waveNumberK[int(j) - 1, s]) * Z[int(j), s])

    Rho_apparent = np.zeros(S)
    Phase = np.zeros(S)
    for s in range(S):
        Rho_apparent[s] = abs(Z[0, s]) ** 2 / (w[s] * mu0)
        Phase[s] = cmath.phase(Z[0, s]) * 180 / cmath.pi

    return Z, Rho_apparent, Phase


def plot_results(w, Rho_apparent, Phase):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.semilogx(1 / w, Rho_apparent, 'b.-')
    ax1.semilogy()
    ax1.set_ylabel('Rho')
    ax1.set_title('Apparent Resistivity')
    ax1.grid(True)

    ax2.semilogx(1 / w, Phase, 'r.-')
    ax2.set_xlabel('Sqrt(T)')
    ax2.set_ylabel('Phi')
    ax2.set_title('Phase Response')
    ax2.grid(True)

    plt.show()


# Example usage
mu0 = 4 * np.pi * 1e-7
w = np.array(progression)  # Frequencies (Hz)
sigma = np.array([0.1, 0.001])  # Conductivities (S/m)
h = np.array([100, inf])  # Layer thicknesses (m)

Z, Rho_apparent, Phase = Fwd1DProblem(mu0, w, sigma, h)

plot_results(w, Rho_apparent, Phase)
