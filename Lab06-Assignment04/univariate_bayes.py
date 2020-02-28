"""
Module defining a base class implementing simple Bayesian inference for a
univariate model, using quadrature for integration.

Created Feb 27, 2015 by Tom Loredo
2018-03-08 Modified for Py-3, updated for BDA18
2020-02-26 Updated for BDA20
"""

import numpy as np
from numpy import *
from matplotlib.pyplot import plot


class UnivariateBayesianInference(object):
    """
    Implement Bayesian inference for a univariate model, using quadrature for
    integrals.
    """

    def __init__(self, param_grid, prior, lfunc=None, llfunc=None):
        """
        Calculate the posterior distribution over a grid in parameter space.

        Either the likelihood or the log-likelihood may be specified.

        Parameters
        ----------
        param_grid : float array
            Array of parameter values; assumed equally spaced

        prior : float or function
            Prior PDF for the param, as a constant for flat prior, or
            a function that can evaluate the PDF on an array

        lfunc : function
            Function that can evaluate the likelihood on an array

        llfunc : function
            Function that can evaluate the log-likelihood on an array
        """
        self.param_grid = param_grid
        self.delta = param_grid[1] - param_grid[0]

        # Evaluate prior and likelihood over the grid.
        if callable(prior):
            self.prior_pdf = prior(param_grid)
        else:
            self.prior_pdf = prior * ones_like(param_grid)
        if lfunc:
            if llfunc:
                raise ValueError('Cannot specify both lfunc & llfunc!')
            self.like = lfunc(param_grid)
        else:  # *** This option is so far untested!
            if llfunc is None:
                raise ValueError('Must specify either lfunc or llfunc!')
            llike = llfunc(param_grid)
            self.like = exp(llike)

        # Bayes's theorem, using the trapezoid rule for the marginal likeilhood:
        numer = self.prior_pdf * self.like
        self.mlike = np.trapz(numer, dx=self.delta)
        self.post_pdf = numer / self.mlike

    def plot(self, ls='b-', lw=3, **kwds):
        """
        Plot the posterior PDF in the current axes.
        """
        plot(self.param_grid, self.post_pdf, ls, lw=lw, **kwds)
