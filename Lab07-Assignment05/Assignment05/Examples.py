"""
Example inference instances for Assignment 05.

2018-03-16:  Revised by Tom Loredo for BDA18
2020-03-05:  Revised for BDA20
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats

from poisson_binomial_cauchy import PoissonRateInference, BinomialInference, \
    CauchyLocationInference


ion()  # for interactive use in a terminal session


def g_mean(params):
    """
    Return the function whose expectation gives the posterior mean, i.e.,
    just return the values of the params.
    """
    params = asarray(params)
    return params


have_laplace = False


#-------------------------------------------------------------------------------
# 1st case:  Poisson, const prior, (n,T) = (16, 2)

r_u = 20.  # upper limit for PDF calculation and plotting

# Create a PRI instance and plot the PDF.
prior_l, prior_u = 0., 1e5
flat_pdf = 1./(prior_u - prior_l)
n, T = 16, 2
pri = PoissonRateInference(T, n, flat_pdf, r_u)
pri.plot(alpha=.5)

xlabel(r'Rate (s$^{-1}$)')
ylabel('PDF (s)')
title('Poisson case')

if have_laplace:
    # Laplace approx for the marg. like. and the mean:
    ampl, locn, sig, ml = pri.laplace()
    laplace_mean = pri.laplace(g_mean)
    post_mean_l = laplace_mean[3]/ml

    # Use results to plot a Gaussian PDF here.

    # Print using string formatting:
    print('Poisson case:')
    print('Marg. like.: {:.4e} (quad), {:.4e} (Laplace)'.format(pri.mlike, ml))
    print('Posterior mean: {:4.2f} (quad), {:4.2f} (Laplace)'.format(pri.post_mean, post_mean_l))
    print()

#-------------------------------------------------------------------------------
# 2nd case:  Binomial, const prior, (n, n_trials) = (8, 12)

# Define the data.
n, n_trials = 8, 12

bi = BinomialInference(n, n_trials)
bfig = figure()  # separate figure for binomial case
bi.plot(alpha=.5)

xlabel(r'$\alpha$')
ylabel('Posterior PDF')
title('Binomial case')

if have_laplace:
    # Laplace approx for the marg. like. and the mean:
    laplace_ml = bi.laplace()
    laplace_mean = bi.laplace(g_mean)
    post_mean_l = laplace_mean[3]/laplace_ml[3]

    # Use results to plot a Gaussian PDF here.

    # Print using string formatting:
    print('Beta case:')
    print('Marg. like.: {:10.4e} (quad), {:10.4e} (Laplace)'.format(bi.mlike, laplace_ml[3]))
    print('Posterior mean: {:4.2f} (quad), {:4.2f} (Laplace)'.format(bi.post_mean, post_mean_l))
    print()

#-------------------------------------------------------------------------------
# 3rd case:  Cauchy, const prior


x0, d = 5., 3.
data = stats.cauchy(x0, d).rvs(5)
flat_pdf = .001  # e.g., for prior range 1e3

cli = CauchyLocationInference(d, data, flat_pdf, (-15., 25.))
cfig = figure()
cli.plot(alpha=.5)
# xlim(-10, 15.)

xlabel('$x_0$')
ylabel('Posterior PDF')
title('Cauchy case; CDF method')

samps = []
for i in range(10000):
    samps.append(cli.samp_cdf())
samps = array(samps)

hist(samps, 50, density=True, color='g', alpha=.5)

if have_laplace:
    # Laplace approx for the marg. like. and the mean:
    ampl, locn, sig, ml = cli.laplace()
    laplace_mean = cli.laplace(g_mean)
    post_mean_l = laplace_mean[3]/ml

    # Use results to plot a Gaussian PDF here.

    # Print using string formatting:
    print('Cauchy case:')
    print('Marg. like.: {:.4e} (quad), {:.4e} (Laplace)'.format(cli.mlike, ml))
    print('Posterior mean: {:4.2f} (quad), {:4.2f} (Laplace)'.format(cli.post_mean, post_mean_l))
