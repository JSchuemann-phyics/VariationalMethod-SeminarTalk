"""
This project implements the variational method for finding an upper bound for the groundstate energy of 1D quantum systems utilizing
importance sampling Monte-Carlo integration. For further reading see the accompanying presentation slides or [Pottorf et al.; Eur. J. Phys. 20 (1999) 205].
"""

from HO import HO
from  poeschl_teller import poeschl_teller
from hydrogen import hydrogen

#harmonic oscillator
HO()

#poeschl-teller potential
poeschl_teller()

#hydrogen atom
hydrogen()




