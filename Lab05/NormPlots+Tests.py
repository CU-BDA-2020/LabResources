"""
Lab05 example script:
    * Plot normal distributions and histogram of samples
    * Integrate simple functions using trapezoid rule
    * Implement unit tests for pytest

2018-02-26 by Tom Loredo
2020-02-20:  Revised for BDA20
"""

import matplotlib.pyplot as plt
from scipy import *
from scipy import stats, integrate

# For unit test assertions:
import numpy.testing as npt


plt.ion()  # interactive mode on


#------------------- Plot normals and samples --------------------------------

# Values of the abscissa variable, mu:
mus = linspace(-1., 6., 200)

# norm(x, y) has x=mean, y=sig (not variance!)
narrow_normal = stats.norm(4., .5)

plt.figure(figsize=(9, 7))
plt.plot(mus, narrow_normal.pdf(mus), 'b-', lw=2, label='Narrow')


wide_normal = stats.norm(2., 3.)
plt.plot(mus, wide_normal.pdf(mus), 'r:', lw=2, label='Wide')

plt.legend()

plt.xlabel(r'$\mu$')
plt.ylabel(r'$p(\mu)$')


# Generate some observations:

N = 1000
mu_samples = narrow_normal.rvs(N)

# Use a *normalized* histogram, for comparison with the PDF.
plt.hist(mu_samples, normed=True)


#-------------------- Integration & unit testing -----------------------------

# A unit test example (not related to the normal distn):

def lin(x):
    """
    A simple linear function, for testing numerical integration.
    """
    return x


def sqr(x):
    """
    A simple quadratic function, for testing numerical integration.
    """
    return x**2

# Integrate over this region, using just a few points.
lo, hi = 1., 2.
xvals = linspace(lo, hi, 5)

# The analytical answers:
lin_expect = hi**2/2. - lo**2/2.
sqr_expect = hi**3/3. - lo**3/3.

# Naive tests using the trapezoid rule

def test_trapz_lin():
    """
    Test trapezoid rule quadrature for a linear function.
    """
    assert integrate.trapz(lin(xvals), xvals) == lin_expect


def test_trapz_sqr():
    """
    Test trapezoid rule quadrature for an x**2 function.
    """
    assert integrate.trapz(sqr(xvals), xvals) == sqr_expect


def test_trapz_sqr_approx():
    """
    Test trapezoid rule quadrature for an x**2 function, recognizing
    we can't expect exact matching of the analytical result.  Require
    1% relative accuracy.
    """
    npt.assert_allclose(integrate.trapz(sqr(xvals), xvals), sqr_expect, rtol=.01)


