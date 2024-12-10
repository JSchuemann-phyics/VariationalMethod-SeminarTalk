from scipy.optimize import minimize
from metropolis_algorithm import metropolis_algorithm
import numpy as np


def evaluate_state(N, M, E_l, psi, delta, x0, lam):
    """
    computes the energy for a given variational parameter lam
    :param N: number of ensembles
    :param M: ensemble size
    :param E_l: local energy
    :param psi: wavefunction
    :param delta: maximum stepsize in the metropolis algorithm
    :param x0: initial sample point in the metropolis algorithm
    :param lam: varational parameter
    :return: energy, standard derivation, list of energy corresponding to each ensemble
    """
    # probability distribution
    p = lambda x: psi(x, lam) * psi(x, lam)

    # create N ensembles of size M (this part is not very memory efficient but allows for array magic instead of nested for-loops)
    ensembles = np.empty((N, M))
    for i in range(N):
        new_ensemble = np.array(metropolis_algorithm(M, p, delta, x0))
        x0 = new_ensemble[-1]
        ensembles[i, :] = new_ensemble

    ensemble_energy_matrix = E_l(ensembles, lam)

    # energy avergage of each ensemble
    ensemble_energy_avg = [sum(ensemble_energy_matrix[i, :]) / M for i in range(N)]

    # energy squared average of each ensemble
    ensemble_energy_sq_avg = [sum(ensemble_energy_matrix[i, :] * ensemble_energy_matrix[i, :]) / M for i in range(N)]

    # energy average over all ensemble
    energy_avg = sum(ensemble_energy_avg) / N

    # energy squared
    energy_sq_avg = sum(ensemble_energy_sq_avg) / N

    # standard deviation
    sigma = np.sqrt((energy_sq_avg - energy_avg ** 2) / (M * (N - 1)))

    return energy_avg, sigma, ensemble_energy_avg