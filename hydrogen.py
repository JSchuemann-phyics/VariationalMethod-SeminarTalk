import numpy as np
from MC_groundstate_variationalMethod import evaluate_state
import matplotlib.pyplot as plt
from metropolis_algorithm import metropolis_algorithm

def hydrogen():

    #hydrogen atom (radial part), exact solution: E0 = -1/2, lam = 1

    def psi_exponential(r, lam):
        if r < 0:
            return 0
        return np.exp(-lam * r)

    # while the wavefunction in spherical coordinates is e^-lam * r the associated
    # probability density is r^2 * e^-(2*lam*r). The evalatue_State() routine samples with
    # psi * psi, therefore we  have to multiply with r here

    def psi_times_r(r, lam):
        return psi_exponential(r, lam) * r


    # #best possible solution with lorentz ansatz: E0 = -4/pi^2, lam = pi/4
    # def psi_lorentz(r, lam):
    #     if r < 0:
    #         return 0
    #     return 1 / (lam + r)
    # def psi_times_r(r, lam):
    #   return psi_lorentz(r, lam) * r


    E_local = lambda r, lam: lam / r - 0.5 * lam * lam - 1/r

    #compute the energy for a range of variational parameters lambda
    N = 200
    M = 500
    delta = 3
    r0 = 0.2
    lam_list = np.linspace(0, 2, 50)
    energy_list = []
    sigma_list = []

    for lam in lam_list:
        energy, sigma, ensemble_energy_avg = evaluate_state(N, M, E_local, psi_times_r, delta, r0, lam)
        energy_list.append(energy)
        sigma_list.append(sigma)

    # plot

    # Customize font sizes
    plt.rc('font', size=14)  # Default text size
    plt.rc('axes', titlesize=25)  # Axes title size
    plt.rc('axes', labelsize=20)  # Axes label size
    plt.rc('xtick', labelsize=20)  # X-axis tick size
    plt.rc('ytick', labelsize=20)  # Y-axis tick size
    plt.rc('legend', fontsize=18)  # Legend font size
    plt.rc('figure', titlesize=22)  # Figure title size

    wcprimary = (78 / 255, 42 / 255, 132 / 255)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    # energy
    ax[0].plot(lam_list, energy_list, "o", color=wcprimary)
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel("Energy")

    # standard derivation
    ax[1].plot(lam_list, sigma_list, "o", color=wcprimary)
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$\sigma$")
    fig.suptitle("Hydrogen atom with exponentially decaying trial function.")

    ax[0].set_facecolor('#ebebeb')
    ax[1].set_facecolor('#ebebeb')

    ax[0].set_title("Variational energy", fontsize=22)
    ax[1].set_title("Standard deviation", fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(hspace=1)

    # ax[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # plot the exact groundstate energy
    ax[0].axhline(y=-0.5, xmin=0, xmax=1, linestyle="--", color="orange", label=r"$E_0$")
    ax[0].legend(loc="upper left")

    ax[0].set_ylim(-0.52, )

    #plt.savefig(fname="hydrogen-plot")
    plt.show()

    return

if __name__ == "__main__":
    hydrogen()