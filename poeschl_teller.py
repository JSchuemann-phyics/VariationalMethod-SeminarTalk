import numpy as np
from MC_groundstate_variationalMethod import evaluate_state

import matplotlib.pyplot as plt

def poeschl_teller():

    #poeschl_teller potential: exact groundstate energy: -1/2, exact groundstate is a legendre functions... let's try gauß

    psi_gauß = lambda x, lam: np.exp(-lam * x * x)
    c = 1 #free parameter in the poeschl-teller potential, should'nt effect the groundstate energy
    E_local = lambda x, lam: lam - 2 * lam * lam * x * x - 0.5 * c * (c+1) / np.cosh(x) / np.cosh(x)


    #compute the energy for a range of variational parameters lambda
    N = 200
    M = 500
    delta = 5
    x0 = 0.2
    lam_list = np.linspace(0.1, 1, 50)
    energy_list = []
    sigma_list = []

    for lam in lam_list:
        energy, sigma, ensemble_energy_avg = evaluate_state(N, M, E_local, psi_gauß, delta, x0, lam)
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
    fig.suptitle("Pöschl-Teller potential with Gaussian trial function")

    ax[0].set_facecolor('#ebebeb')
    ax[1].set_facecolor('#ebebeb')

    ax[0].set_title("Variational energy", fontsize=22)
    ax[1].set_title("Standard deviation", fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(hspace=1)

    #ax[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    #plot the exact groundstate energy
    ax[0].axhline(y = -0.5, xmin = 0, xmax = 1, linestyle = "--", color = "orange", label = r"$E_0$")
    ax[0].legend(loc = "upper left")

    ax[0].set_ylim(-0.52,)
    #plt.savefig(fname="pt-plot")
    plt.show()

    E0 = min(energy_list)
    lam_min = lam_list[energy_list.index(E0)]
    print("______________________")
    print("Poeschl-Teller solution:")
    print("groundstate energy: ", E0)
    print("minimizing variational parameter: ", lam_min)
    print("______________________")

    return
if __name__ == "__main__":
    poeschl_teller()