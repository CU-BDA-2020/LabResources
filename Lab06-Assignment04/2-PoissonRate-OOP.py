"""
Plot Poisson rate posterior distributions for multiple datasets.

This implementation uses the object oriented programming (OOP) paradigm.

Created Feb 27, 2015 by Tom Loredo
2018-03-08 Modified for Py-3, updated for BDA18
2020-02-26 Updated for BDA20
"""

import numpy as np
import numpy.testing as npt
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats


ion()


class PoissonRateInference:
    """
    Bayesian inference for a Poisson rate.
    """

    def __init__(self, intvl, n, prior, r_u, r_l=0, nr=200):
        """
        Define a posterior PDF for a Poisson rate.

        Parameters
        ----------
        intvl : float
            Interval for observations

        n : int
            Counts observed

        prior : const or function
            Prior PDF for the rate, as a constant for flat prior, or
            a function that can evaluate the PDF on an array

        r_u : float
            Upper limit on rate for evaluating the PDF
        """
        self.intvl = intvl
        self.r_l, self.r_u = r_l, r_u
        self.nr = nr
        self.rvals = linspace(r_l, r_u, nr)
        self.r_intvl = intvl * self.rvals

        # Evaluate the prior on the grid.
        self.prior = prior  # save for possible future reference
        if callable(prior):
            self.prior_pdf = prior(self.rvals)
        else:
            self.prior_pdf = prior * ones_like(self.rvals)

        # Evaluate the Poisson likelihood function.
        self.like = (self.r_intvl)**n * exp(-self.r_intvl)

        # *** PAY NO ATTENTION TO THE CODE BEHIND THE CURTAIN! ***
        # llike = n*log(self.r_intvl) - self.r_intvl
        # self.like = exp(llike)

        # Bayes's theorem:
        numer = self.prior_pdf * self.like
        self.dr = self.rvals[1] - self.rvals[0]
        self.mlike = np.trapz(numer, dx=self.dr)
        self.post_pdf = numer / self.mlike

    def plot(self, ls='b-', lw=3, **kwds):
        """
        Plot the posterior PDF in the current axes.
        """
        plot(self.rvals, self.post_pdf, ls, lw=lw, **kwds)


#-------------------------------------------------------------------------------
# 1st case:  const prior, (n,T) = (16, 2)

r_u = 20.  # upper limit for PDF calculation and plotting
prior_l, prior_u = 0., 1e5
flat_pdf = 1. / (prior_u - prior_l)
n, T = 16, 2
pri1 = PoissonRateInference(T, n, flat_pdf, r_u)
pri1.plot(alpha=.5)

xlabel(r'Rate (s$^{-1}$)')
ylabel('PDF (s)')


def test_norm1():
    """
    Test that the posterior is normalized.
    """
    # Match 1 to 2 digits.
    npt.assert_allclose(np.trapz(pri1.post_pdf, dx=pri1.dr), 1., rtol=0.01)


#-------------------------------------------------------------------------------
# 2nd case:  exp'l prior with scale (prior mean) 10., (n,T) = (16, 2)

# Prior:
scale = 10.
gamma1 = stats.gamma(1, scale=scale)  # a=1 is exp'l dist'n

pri2 = PoissonRateInference(T, n, gamma1.pdf, r_u)
pri2.plot(ls='g--')


#-------------------------------------------------------------------------------
# 3rd case:  flat prior with (n,T) = (80, 10)

n, T = 80, 10.  # data

pri3 = PoissonRateInference(T, n, flat_pdf, r_u)
pri3.plot(alpha=.5)


#-------------------------------------------------------------------------------
# 4th case:  exp'l prior with scale (prior mean) 10., (n,T) = (80, 10)

pri4 = PoissonRateInference(T, n, gamma1.pdf, r_u)
pri4.plot(ls='g--')


def test_norm4():
    """
    Test that the posterior is normalized.
    """
    # Match 1 to 2 digits.
    npt.assert_allclose(np.trapz(pri4.post_pdf, dx=pri1.dr), 1., rtol=0.01)


#-------------------------------------------------------------------------------
# 5th & 6th cases:  (n,T) = (160, 20)

if False:
    n, T = 160, 20.  # data

    pri5 = PoissonRateInference(T, n, flat_pdf, r_u)
    pri5.plot(alpha=.5)
    pri6 = PoissonRateInference(T, n, gamma1.pdf, r_u)
    pri6.plot(ls='g--')
