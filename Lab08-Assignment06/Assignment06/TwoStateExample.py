"""
Demonstrate use of the TwoStateMarkovChain class.

Modified for BDA18 2018-03-23 by Tom Loredo
2020-03-10:  Updated for BDA20
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats

# try:
#     import myplot
# except ImportError:
#     pass

from two_state_markov import TwoStateMarkovChain


ion()


# Define two initializers --- samplers for the starting state.

# Deterministic (Kronecker delta) initial state distribution:
def init_at_0():
    """
    Initial state sampler returning state 0.
    """
    return 0


# Fair coin flip random initial state distribution:
init_half = stats.binom(1, .5).rvs


# Instantiate a chain generator.
# Experiment with (alpha, beta) to control convergence to equilibrium.
tsmc = TwoStateMarkovChain(.07, .03)

n_paths = 20  # number of sample paths to generate

# Experiment with n_paths, alpha_trace and dither:
# * The more paths, the lower alpha_trace should be
# * With nonzero dither, less transparency is needed (alpha can be greater)

# Simulate paths of length 30, all starting at 0.
tsmc.sim_paths(n_paths, init_at_0, 30)
tsmc.plot_evol(alpha_trace=.08, dither=0.)


# Simulate paths of length 60, starting randomly (p = 1/2).
tsmc.sim_paths(n_paths, init_half, 60)
tsmc.plot_evol(alpha_trace=.15, dither=.05)
