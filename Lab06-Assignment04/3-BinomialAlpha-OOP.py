"""
Plot posterior distributions for a binomial parameter.

This implementation uses the object oriented programming (OOP) paradigm.

Created Feb 27, 2015 by Tom Loredo
2018-03-08 Modified for Py-3, updated for BDA18
2020-02-26 Updated for BDA20
"""

import numpy as np
# import numpy.testing as npt
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats, special, integrate


ion()


class BinomialInference:
    """
    Bayesian inference for the probability of a Bernoulli outcome, based
    on binomial data.
    """

    def __init__(self, n, n_trials, prior=1., na=200, arange=(0., 1.)):
        """
        Define a posterior PDF for the probability of a Bernoulli outcome,
        alpha, based on binomial data.

        Parameters
        ----------

        n : int
            Number of successes

        n_trials : int
            Number of trials (>= n)

        prior : const or function
            Prior PDF for alpha, as a constant for flat prior, or
            a function that can evaluate the PDF on an array
        """
        self.n, self.n_trials = n, n_trials
        self.na = na
        self.alphas = linspace(arange[0], arange[1], na)

        # Evaluate the prior on the grid.
        self.prior = prior  # save for possible future reference
        if callable(prior):
            self.prior_pdf = prior(self.alphas)
        else:
            self.prior_pdf = prior * ones_like(self.alphas)

        # Evaluate the binomial likelihood function; ignore the
        # combinatorial factor (indep. of alpha).
        self.like = (self.alphas)**n * (1. - self.alphas)**(n_trials - n)

        # Bayes's theorem:
        numer = self.prior_pdf * self.like
        self.da = self.alphas[1] - self.alphas[0]
        self.mlike = np.trapz(numer, dx=self.da)
        self.post_pdf = numer / self.mlike

    def plot(self, ls='b-', lw=3, **kwds):
        """
        Plot the posterior PDF in the current axes.
        """
        plot(self.alphas, self.post_pdf, ls, lw=lw, **kwds)


#-------------------------------------------------------------------------------
# 1st 2 cases:  const & beta(.5,.5) priors, (n, n_trials) = (8, 12)

# Define the data.
n, n_trials = 8, 12

bi1 = BinomialInference(n, n_trials)
bi1.plot(alpha=.5)

beta_half = stats.beta(a=.5, b=.5)
bi2 = BinomialInference(n, n_trials, beta_half.pdf, arange=(1.e-4, 1 - 1.e-4))
bi2.plot(ls='g--')

xlabel(r'$\alpha$')
ylabel('Posterior PDF')


#-------------------------------------------------------------------------------
# 2nd 2 cases:  const & beta(.5,.5) priors, (n, n_trials) = (4, 12)

n, n_trials = 4, 12

bi3 = BinomialInference(n, n_trials)
bi3.plot(alpha=.5)

bi4 = BinomialInference(n, n_trials, beta_half.pdf, arange=(1.e-4, 1 - 1.e-4))
bi4.plot(ls='g--')


#-------------------------------------------------------------------------------
# 3rd 2 cases:  const & beta(.5,.5) priors, (n, n_trials) = 4*(8, 12)

n, n_trials = 4 * 8, 4 * 12

bi5 = BinomialInference(n, n_trials)
bi5.plot(alpha=.5)

bi6 = BinomialInference(n, n_trials, beta_half.pdf, arange=(1.e-4, 1 - 1.e-4))
bi6.plot(ls='g--')
