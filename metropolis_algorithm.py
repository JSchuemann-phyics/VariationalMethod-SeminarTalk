import numpy as np
import matplotlib.pyplot as plt
def metropolis_algorithm(N, p, delta, x0):
    """
    Generates list samples, which values follow the distribution p(x)
    :param N: Number of generated point
    :param p: probability distribution
    :param delta: maximum step size
    :param x0: first trial point
    :return: samples, list of random values that follow p(x)
    """

    accepted = 0
    tried = 0
    x_old = x0
    samples = []
    while tried < N:
        tried += 1
        step = np.random.uniform(-delta, delta)
        x_new = x_old + step

        w = p(x_new)/p(x_old)
        r = np.random.uniform(0, 1)

        #acceptance case
        if w >= r:
            samples.append(x_new)
            x_old = x_new
            accepted += 1
        else:
            samples.append(x_old)
    #print("acceptance rate:", accepted/tried)
    return samples

##test
if __name__ == "__main__":
    def gauß(x):
        return np.exp(-0.5 * x * x) * 1/np.sqrt(2 * np.pi)

    def exponential(x):
        if x < 0:
            return 0
        return np.exp(-x)

    def linear(x):
        if x < 0 or x > 1:
            return 0
        return 2 * (1 - x)
    x0 = 0
    delta = 3
    N_eq = 1000 # samples to reach equilibrium
    N = 1e5
    p = gauß #choose a function to test with

    samples = metropolis_algorithm(N_eq, p, delta, x0)
    x0 = samples[-1]
    samples = metropolis_algorithm(N, p, delta, x0)

    binwidth = 0.01

    fig, ax = plt.subplots(1, 1)
    ax.hist(np.array(samples), bins=np.arange(min(samples), max(samples) + binwidth, binwidth), density=True)
    p = np.vectorize(p)
    x = np.linspace(-5, 5, int(N))
    ax.plot(x, p(x))
    plt.show()



