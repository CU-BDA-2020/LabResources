"""
Helper objects for inference of a straight-line regression function.

Created Apr 21, 2015 by Tom Loredo
2020-04-24:  Modified for BDA20, including support of vector fits
"""

import numpy as np
import scipy
from numpy import *
from scipy import stats
from matplotlib.pyplot import *


class MargCondJoint2D:

    def __init__(self, x_marg, y_cond_x):
        """
        Define a joint dist'n for (x,y) by specifying a marginal for x and
        a conditional for y given x.

        Parameters
        ----------

        x_marg : distribution instance
            An object defining the marginal, p(x); it should have a pdf(xvals)
            method evaluating the marginal PDF, and an rvs(n) method that
            returns n samples

        y_cond_x : conditional distribution provider function
            An function returning the conditional, p(y|x), as a function of x;
            y_cond_x(x) should return a distribution instance with the same
            required methods as x_marg
        """
        self.x_marg = x_marg
        self.y_cond_x = y_cond_x

    def pdf(self, xy):
        """
        Return the joint PDF, p(x,y), for an (x,y) point:
            xy[0] = x
            xy[1] = y
        """
        y_cond = self.y_cond_x(xy[0])
        return self.x_marg.pdf(xy[0]) * y_cond.pdf(xy[1])

    def sample(self, n=1):
        """
        Return `n` samples from the joint distribution as two
        arrays, of x and y coordinates.
        """
        if n==1:
            xvals = array( [self.x_marg.rvs()] )
        else:
            xvals = self.x_marg.rvs(n)
        yvals = empty_like(xvals)
        for i, x in enumerate(xvals):
            y_cond = self.y_cond_x(x)
            yvals[i] = y_cond.rvs()
        return xvals, yvals



def traceplots(fit, vector=False):
    """
    Make a set of traceplots for StanFitResults instance `fit`.
    
    Use `vector=False` if the fit uses scalar coefficients, and
    `vector=True` if it uses a coefficient vector.
    """
    f=figure(figsize=(10,9))
    ax=f.add_subplot(3,1,1)
    if not vector:
        # Note: without `axes`, this would make its own fig.
        fit.beta_0.trace(axes=ax,alpha=.6)
    else:
        fit.beta[0].trace(axes=ax,alpha=.6)
    ax.set_xlabel('')

    ax=f.add_subplot(3,1,2)
    if not vector:
        fit.beta_1.trace(axes=ax,alpha=.6)
    else:
        fit.beta[1].trace(axes=ax,alpha=.6)
    ax.set_xlabel('')

    ax=f.add_subplot(3,1,3)
    fit.log_p.trace(axes=ax,alpha=.6)


def betas_plot(fit, beta_0, beta_1, vector=False):
    """
    Make a scatterplot of the beta samples from StanFitResults instance `fit`.

    Draw a crosshair showing true values `beta_0` and `beta_1`.
    
    This supports `fit` objects using scalar params `fit.beta_0` and `fit.beta_1`,
    or vector params in `fit.beta`, depending on the `vector` argument.
    """
    # Pull out needed info depending on whether a vector
    # parameterization was used or not.
    nsc = fit.chains.shape[0]  # number of samples kept per chain
    if not vector:
        # Thin samples using smallest ESS.
        thin_by = nsc / min(fit.beta_0.ess, fit.beta_1.ess)
        beta_0_fit, beta_1_fit = fit.beta_0, fit.beta_1
    else:  # vector case
        thin_by = nsc / min(fit.beta[0].ess, fit.beta[1].ess)
        beta_0_fit, beta_1_fit = fit.beta[0], fit.beta[1]

    # Thin by an *integer*:
    thin_by = int(ceil(thin_by))
    print('Thinning chains by', thin_by)

    figure()
    xlabel(r'$\beta_0$')
    ylabel(r'$\beta_1$')

    # Go through chains.
    nc = fit.chains.shape[1]
    for c in range(nc):
        b0 = beta_0_fit.chains[::thin_by,c]
        b1 = beta_1_fit.chains[::thin_by,c]
        scatter(b0, b1, s=15, c='b', linewidths=0, alpha=.2)

    # Thin-lined crosshair showing true values:
    xhair = dict(c='k', ls=':', lw=1.5, alpha=.5)
    axvline(beta_0, **xhair)
    axhline(beta_1, **xhair)

