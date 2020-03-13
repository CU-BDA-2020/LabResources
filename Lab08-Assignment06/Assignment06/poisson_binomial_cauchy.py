"""
Plot Poisson rate posterior PDFs, and binomial alpha posterior PDFs, as
a demo of the UnivariateBayesianInference class.

Created Feb 27, 2015 by Tom Loredo
2020-03-05:  Revised for BDA20
"""

from numpy import *

from univariate_bayes import UnivariateBayesianInference


__all__ = ['BinomialInference', 'PoissonRateInference']


class BinomialInference(UnivariateBayesianInference):
    """
    Bayesian inference for the probability of a Bernoulli outcome, based
    on binomial data.
    """

    def __init__(self, n, n_trials, prior=1., na=200, arange=(0., 1.)):
        """
        Define a posterior PDF for the probability of a Bernoulli outcome,
        alpha, based on binomail data.

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

        # Pass info to the base class initializer.
        super(BinomialInference, self).__init__(self.alphas, prior, self.lfunc)

    def lfunc(self, alphas):
        """
        Evaluate the Binomial likelihood function on a grid of alphas.
        """
        # Ignore the combinatorial factor (indep. of alpha).
        return (alphas)**self.n * (1.-alphas)**(self.n_trials - self.n)


class PoissonRateInference(UnivariateBayesianInference):
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
        self.n = n
        self.r_l, self.r_u = r_l, r_u
        self.nr = nr
        self.rvals = linspace(r_l, r_u, nr)

        # Pass info to the base class initializer.
        super(PoissonRateInference, self).__init__(self.rvals, prior, self.lfunc)

    def lfunc(self, rvals):
        """
        Evaluate the Poisson likelihood function on a grid of rates.
        """
        r_intvl = self.intvl*rvals
        return (r_intvl)**self.n * exp(-r_intvl)


class CauchyLocationInference(UnivariateBayesianInference):
    """
    Bayesian inference for the location parameter of the Cauchy dist'n.
    """

    def __init__(self, scale, data, prior, x0_range=None, n=250):
        """
        Define a posterior PDF for the location parameter, x0, of a
        Cauchy dist'n.

        Parameters
        ----------
        scale : float
            Cauchy scale parameter (presumed known)

        data : float array
            Vector of samples modeled as from a Cauchy dist'n

        prior : const or function
            Prior PDF for the location, as a constant for flat prior, or
            a function that can evaluate the prior PDF on an array

        x0_range : 2-tuple of floats
            Range of x0 defining the grid over which the posterior PDF will
            be evaluated; if None, the range of the data is used

        n : int
            Number of x0 values on the grid
        """
        self.scale = scale
        self.scale2 = scale*scale
        self.data = data
        if x0_range is None:
            self.x0_l = data.min()
            self.x0_u = data.max()
        else:
            self.x0_l, self.x0_u = x0_range
        self.n = n
        self.x0_grid = linspace(self.x0_l, self.x0_u, n)

        # Pass info to the base class initializer.
        super(CauchyLocationInference, self).__init__(self.x0_grid, prior, self.lfunc)

    def lfunc(self, x0vals):
        """
        Evaluate the Cauchy likelihood function for x0vals (scalar or vector).
        """
        x0vals = asarray(x0vals)  # gives scalars .shape
        if len(x0vals.shape) == 0:  # scalar argument -> scalar result
            return prod(1./(1. + (x0vals - self.data)**2/self.scale2))
        # Handle vector of x0vals here.
        lvals = empty_like(x0vals)
        for i in range(len(x0vals)):
            lvals[i] = prod(1./(1. + (x0vals[i] - self.data)**2/self.scale2))
        return lvals
