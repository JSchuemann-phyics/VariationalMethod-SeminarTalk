import numpy as np
from MC_groundstate_variationalMethod import evaluate_state
from mc_integration import mc_integration

import matplotlib.pyplot as plt

def HO():
    #harmonic oszillator, exact results: lam = 0.5, E0 = 0.5
    psi_gauß = lambda x, lam: np.exp(-lam * x * x)
    E_local = lambda x, lam: lam - 2 * lam * lam * x * x + 0.5 * x * x
    psi_H_psi = lambda x, lam: np.sqrt(2 * lam / np.pi) * np.exp(-2 * lam * x * x) * E_local(x, lam) #used in normal MC for comparison

    #compare (normal) Monte-Carlo method with importance sampling for some variational parameter
    N = 1000
    M = 500
    lam = 0.5

    #for (normal) MC
    L = 10 #range for normal MC
    x_span = (-L, L)
    f = lambda x: psi_H_psi(x, lam)

    #for importance sampling
    delta = 2.6
    x0 = 0

    #importance sampling
    energy_IS, sigma_IS, ensemble_energy_avg_IS = evaluate_state(N, M, E_local, psi_gauß, delta, x0, lam)


    #simple sampling
    energy_MC, sigma_MC, ensemble_energy_avg_MC = mc_integration(N, M, f, x_span)

    binwidth = 0.01
    alpha = 0.7

    plt.style.use('default')

    # Customize font sizes
    plt.rc('font', size=14)  # Default text size
    plt.rc('axes', titlesize=25)  # Axes title size
    plt.rc('axes', labelsize=20)  # Axes label size
    plt.rc('xtick', labelsize=20)  # X-axis tick size
    plt.rc('ytick', labelsize=20)  # Y-axis tick size
    plt.rc('legend', fontsize=18)  # Legend font size
    plt.rc('figure', titlesize=22)  # Figure title size


    wcprimary = (78 / 255, 42 / 255, 132 / 255)
    plt.hist(ensemble_energy_avg_MC, bins=np.arange(min(ensemble_energy_avg_MC), max(ensemble_energy_avg_MC) + binwidth, binwidth), color = "orange", label = "Simple sampling", alpha = alpha)
    plt.hist(ensemble_energy_avg_IS, bins=np.arange(min(ensemble_energy_avg_IS), max(ensemble_energy_avg_IS) + binwidth, binwidth), color = wcprimary, label = "Importance sampling", alpha = alpha)
    plt.title(r"Comparison of simple sampling to importance sampling ($\lambda = {}$)".format(lam))
    ax = plt.gca()
    ax.set_xlabel(r"$S_{MC}$")
    ax.set_ylabel(r"Distribution $P(S_{MC})$")
    ax.set_xlim(0.25, 1)

    ax.set_facecolor('#ebebeb')
    plt.legend()

    plt.show()

    #compute energy for a range of varational parameters
    N = 50  #number of ensembles
    M = 500 #ensemble size
    delta = 2.6
    x0 = 0

    lam_list = np.linspace(0.2, 1, 50)
    energy_list = []
    sigma_list = []
    for lam in lam_list:
        energy, sigma, ensemble_energy_avg = evaluate_state(N, M, E_local, psi_gauß, delta, x0, lam)
        energy_list.append(energy)
        sigma_list.append(sigma)

    #plot

    # Customize font sizes
    plt.rc('font', size=14)  # Default text size
    plt.rc('axes', titlesize=25)  # Axes title size
    plt.rc('axes', labelsize=20)  # Axes label size
    plt.rc('xtick', labelsize=20)  # X-axis tick size
    plt.rc('ytick', labelsize=20)  # Y-axis tick size
    plt.rc('legend', fontsize=18)  # Legend font size
    plt.rc('figure', titlesize=22)  # Figure title size

    wcprimary = (78 / 255, 42 / 255, 132 / 255)

    fig, ax = plt.subplots(1, 2, figsize = (15, 7))
    #energy
    ax[0].plot(lam_list, energy_list, "o", color = wcprimary)
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel("Energy")

    #standard derivation
    ax[1].plot(lam_list, sigma_list,"o", color = wcprimary)
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$\sigma$")
    fig.suptitle("Harmonic oscillator with Gaussian trial function")

    ax[0].set_facecolor('#ebebeb')
    ax[1].set_facecolor('#ebebeb')

    ax[0].set_title("Variational energy", fontsize = 22)
    ax[1].set_title("Standard deviation", fontsize = 22)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 1)

    ax[1].ticklabel_format(axis = "both", style="sci", scilimits=(0,0))

    # plot the exact groundstate energy
    ax[0].axhline(y=0.5, xmin=0, xmax=2, linestyle="--", color="orange", label=r"$E_0$")
    ax[0].legend(loc="upper left")

    ax[0].set_ylim(0.48, )
    #plt.savefig(fname = "HO-plot-2")
    #ax[1].remove()
    #plt.savefig(fname = "HO-plot-1")
    plt.show()

    return

if __name__ == "__main__":
    HO()
