import numpy as np
from scipy.optimize import minimize

def groundState_variationalMethod(x,Vx):
    r"""Ground state obtained via variational method

    Minimizes the energy-functional for a Gaussian trial wave function.

    Notes:
    - The second order derivative in the kinetic energy term is approximated
      using a three-point finite difference stencil. Thus, this implementation
      is more versatile since it can easily be used for other parameterized
      trial functions as well.

    Args:
        x (array): discrete x-grid
        Vx (array): potential in array-form (not as a function)

    Returns (s0, E0_var, psi0_var):
        s0 (float): width of the adapted Gaussian trial function
        E0_var (float): ground state energy obtained in terms of the
            variational method
        psi0_var (array): ground state wavefunction obtained in terms of the
            variational method
    """
    dx=x[1]-x[0]

    # -- NORMALIZED TRIAL WAVE FUNCTION
    # ... TRIAL FUNCTION
    fun = lambda s0: np.sqrt(np.exp(-(x)**2/2/s0/s0))
    # ... NORMALIZATION
    psi = lambda s0: fun(s0)/np.sqrt(np.trapz(fun(s0)**2,dx=dx))

    # -- VARIATIONAL INTEGRAL (ENERGY FUNCTIONAL)
    def _E(s0):
        p = psi(s0)
        d2p_dx2 = (p[:-2] - 2*p[1:-1] + p[2:])/dx/dx
        return np.trapz(-0.5*p[1:-1]*d2p_dx2, dx=dx) + np.trapz(p**2*Vx, dx=dx)

    # -- MINIMIZE VARIATIONAL INTEGRAL
    res = minimize(_E, 1., method='Nelder-Mead', tol=1e-6 )
    return res.x, _E(res.x), psi(res.x)
