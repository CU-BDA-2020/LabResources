"""
Plot Poisson rate posterior distributions for multiple datasets.

This implementation uses the procedural programming paradigm.

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


#-------------------------------------------------------------------------------
# 1st case:  const prior, (n,T) = (16, 2)

# Parameters specifying the 1st case with a flat prior:
# Make sure prior is proper by bounding it; this choice should not
# depend on the data---it's a *prior*!
prior_l, prior_u = 0., 1e5
n, T = 16, 2.  # data
# Plot range; this will depend on the data; educated guess:
r_l, r_u = 0., 20.  # n/T plus few*sqrt(n)/T

# Define the grid of rates for computing & plotting.
rvals = linspace(r_l, r_u, 200)
rtvals = rvals * T
dr = rvals[1] - rvals[0]

# Bayes's theorem:
prior_pdf = ones_like(rvals) / (prior_u - prior_l)
# Poisson dist'n likelihood, dropping n! factor:
like = (rtvals)**n * exp(-rtvals)
numer = prior_pdf * like
denom = np.trapz(numer, dx=dr)
post_pdf = numer / denom

# Plot the posterior
plot(rvals, post_pdf, 'b-', lw=3, alpha=0.5)
xlabel(r'Rate (s$^{-1}$)')
ylabel('PDF (s)')


def test_norm1():
    """
    Test that the posterior is normalized.
    """
    # Match 1 to 2 digits.
    npt.assert_allclose(np.trapz(post_pdf, dx=dr), 1., 2)


#-------------------------------------------------------------------------------
# 2nd case:  exp'l prior with scale (prior mean) 10., (n,T) = (16, 2)

# Prior:
scale = 10.
prior = stats.gamma(1, scale=scale)  # a=1 is exp'l dist'n

# Easiest to just copy/paste things that are different this time:

# Bayes's theorem:
numer = prior.pdf(rvals) * like
denom = np.trapz(numer, dx=dr)
post_pdf = numer / denom

# Plot the posterior
plot(rvals, post_pdf, 'g--', lw=3)


#-------------------------------------------------------------------------------
# 3rd case:  flat prior with (n,T) = (80, 10)

n, T = 80, 10.  # data

# rtvals changes because T changed:
rtvals = rvals * T

# Bayes's theorem:
prior_pdf = ones_like(rvals) / (prior_u - prior_l)
# Poisson dist'n likelihood, dropping n! factor:
like = (rtvals)**n * exp(-rtvals)
numer = prior_pdf * like
denom = np.trapz(numer, dx=dr)
post_pdf = numer / denom

plot(rvals, post_pdf, 'b-', lw=3, alpha=0.5)

#-------------------------------------------------------------------------------
# 4th case:  exp'l prior with scale (prior mean) 10., (n,T) = (80, 10)

numer = prior.pdf(rvals) * like
denom = np.trapz(numer, dx=dr)
post_pdf = numer / denom

# Plot the posterior
plot(rvals, post_pdf, 'g--', lw=3)


def test_norm4():
    """
    Test that the posterior is normalized.
    """
    # Match 1 to 2 digits.
    npt.assert_allclose(np.trapz(post_pdf, dx=dr), 1., rtol=0.01)


#-------------------------------------------------------------------------------
# 5th case:  flat prior with (n,T) = (160, 20)

if False:
    n, T = 160, 20.  # data

    # rtvals changes because T changed:
    rtvals = rvals * T

    # Bayes's theorem:
    prior_pdf = ones_like(rvals) / (prior_u - prior_l)
    # Poisson dist'n likelihood, dropping n! factor:
    like = (rtvals)**n * exp(-rtvals)
    numer = prior_pdf * like
    denom = np.trapz(numer, dx=dr)
    post_pdf = numer / denom

    plot(rvals, post_pdf, 'b-', lw=3, alpha=0.5)
