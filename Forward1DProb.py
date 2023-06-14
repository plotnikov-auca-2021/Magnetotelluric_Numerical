import numpy as np
import matplotlib.pyplot as plt
from cmath import tanh, phase


def wave_number_k(sigma, w, mu0):
    J = len(sigma)  # number of layers
    S = len(w)  # number of frequencies

    waveNumberK = np.zeros((J, S), dtype=complex)

    for j in range(J):
        for s in range(S):
            waveNumberK[j, s] = (1 - 1j) * np.sqrt(w[s] * mu0 * sigma[j] / 2.0)

    return waveNumberK


def Fwd1DProblem(mu0, w, sigma, h):
    S = len(w)  # number of frequencies
    J = len(sigma)  # number of layers

    waveNumberK = wave_number_k(sigma, w, mu0)

    Z = np.zeros((J, S), dtype=complex)
    for j in range(J, 0, -1):
        for s in range(S):
            if j == J:
                Z[j - 1, s] = waveNumberK[J - 1, s] / sigma[J - 1]
            else:
                Z[j - 1, s] = (waveNumberK[j - 1, s] * Z[j, s] - 1j * w[s] * mu0 * tanh(
                    h[j - 1] * waveNumberK[j - 1, s])) / \
                              (waveNumberK[j - 1, s] + sigma[j - 1] * tanh(h[j - 1] * waveNumberK[j - 1, s]) * Z[j, s])

    Rho_apparent = np.zeros(S)
    Phase = np.zeros(S)
    for s in range(S):
        Rho_apparent[s] = abs(Z[0, s]) ** 2 / (w[s] * mu0)
        Phase[s] = np.angle(Z[0, s]) * 180 / np.pi

    return Z, Rho_apparent, Phase


def plot_results(w, Rho_apparent, Phase):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

    ax1.semilogx(w, Rho_apparent, 'b.-')
    ax1.set_ylabel('Rho')
    ax1.set_title('Apparent Resistivity and Phase Response')
    ax1.grid(True)

    ax2.semilogx(w, Phase, 'r.-')
    ax2.set_xlabel('Sqrt(T)')
    ax2.set_ylabel('Phi')
    ax2.grid(True)

    plt.show()


# Example usage
mu0 = 4 * np.pi * 1e-7
w = np.array([0.1, 1, 6, 8, 11, 45, 67, 89, 100, 726, 928, 8573, 9234])  # Frequencies (Hz)
sigma = np.array([100, 300, 1000])  # Conductivities (S/m)
h = np.array([100, 200, 100])  # Layer thicknesses (m)

Z, Rho_apparent, Phase = Fwd1DProblem(mu0, w, sigma, h)

plot_results(w, Rho_apparent, Phase)
