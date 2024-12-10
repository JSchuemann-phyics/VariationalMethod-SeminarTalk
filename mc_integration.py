import numpy as np
import matplotlib.pyplot as plt
def mc_integration(N, M, fun, x_span):
   """
   1D-Monte-Carlo integration without importance sampling
   :param N: number of ensembles
   :param M: number of sample point that are used for the integration
   :param fun: function, that is to be integrated
   :param x_span: boundaries, fun is integrated from x_span[0] to x_span[1]
   :return: sol, estimate of the integral
   """
   a = x_span[0]
   b = x_span[1]
   ensembles = np.random.uniform(a, b, (N, M))

   ensemble_energy_matrix = fun(ensembles)

   # energy avergage of each ensemble
   ensemble_energy_avg = [(b - a) / M * sum(ensemble_energy_matrix[i, :]) for i in range(N)]

   # energy squared average of each ensemble
   ensemble_energy_sq_avg = [(b - a)**2 / M * sum(ensemble_energy_matrix[i, :] * ensemble_energy_matrix[i, :]) for i in range(N)]

   # energy average over all ensemble
   energy_avg = 1 / N * sum(ensemble_energy_avg)

   # energy squared
   energy_sq_avg = 1 / N * sum(ensemble_energy_sq_avg)

   # standard deviation
   sigma = np.sqrt((energy_sq_avg - energy_avg ** 2) / (M * (N - 1)))

   return energy_avg, sigma, ensemble_energy_avg

#test
if __name__ == "__main__":
   fun = lambda x: np.sqrt(1 - x * x)
   x_span = (0, 1)
   N = 1000
   M = 500
   sol, sigma, ensemble_sol = mc_integration(N, M, fun, x_span)

   pi = 4 * sol

   print("approximation of pi: ", pi)

   plt.hist(4 * np.array(ensemble_sol), bins = 50)
   plt.show()












