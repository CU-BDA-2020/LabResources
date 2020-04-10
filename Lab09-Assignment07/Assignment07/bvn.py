"""
Module for exploring bivariate normal distributions and their conditional
and marginal distributions.

2018-04-12: Modified for Py-3 and BDA18 by Tom Loredo
2020-04-09: Revised for BDA20
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats
from scipy.stats import multivariate_normal


class BivariateNormal:
    """
    Bivariate normal dist'n, including specification of conditionals and
    marginals.
    """

    def __init__(self, means, sigs, rho):
        """
        Define the BVN via its joint dist'n description in terms of marginal
        means, marginal std deviations, and the correlation coefficient.

        Parameters
        ----------
        means : 2-element float sequence
            Marginal means

        sigs : 2-element float sequence
            Marginal standard deviations

        rho : float
            Correlation coefficient
        """
        self.means = asarray(means)
        self.sigs = asarray(sigs)
        self.rho = rho
        cross = rho*sigs[0]*sigs[1]
        self.cov = array([[sigs[0]**2, cross], [cross, sigs[1]**2]])
        self.bvn = multivariate_normal(self.means, self.cov)

        # Conditional for y|x:
        self.slope_y = rho*sigs[1]/sigs[0]
        self.int_y = means[1] - self.slope_y*means[0]
        self.csig_y = sigs[1]*sqrt(1. - rho**2)

        # Conditional for x|y:
        self.slope_x = rho*sigs[0]/sigs[1]
        self.int_x = means[0] - self.slope_x*means[1]
        self.csig_x = sigs[0]*sqrt(1. - rho**2)

    def y_x(self, x):
        """
        Return the conditional expectation for `y` given `x`.
        """
        return self.int_y + self.slope_y*x

    def x_y(self, y):
        """
        Return the conditional expectation for `x` given `y`.
        """
        return self.int_x + self.slope_x*y

    def pdf(self, xy):
        """
        Evaluate the PDF at the point `xy`.
        """
        return self.bvn.pdf(asarray(xy))

    def sample(self, n=1):
        """
        Return an array of `n` samples from the BVN.
        """
        return self.bvn.rvs(n)

    def xy_grid(self, n, fac=5.):
        """
        Define vectors and 2-D arrays useful for plotting a BVN and related
        functions.
        """
        # Vectors of x and y values:
        x = linspace(self.means[0]-fac*self.sigs[0],
                     self.means[0]+fac*self.sigs[0], n)
        y = linspace(self.means[1]-fac*self.sigs[1],
                     self.means[1]+fac*self.sigs[1], n)
        # 2-D float arrays giving the x or y values over the grid;
        # note that these use 'xy' indexing, so grid element [j,i]
        # corresponds to (x[i], y[j]), i.e., a row is over x values.
        xg, yg = np.meshgrid(x, y)
        # 2-D array of 2-vectors giving (x,y) over the grid:
        xyg = empty(xg.shape + (2,))
        xyg[:,:,0] = xg
        xyg[:,:,1] = yg
        return x, y, xg, yg, xyg


def plot_bvn(ax, bvn, rlines=False, regress=None, samples=None):
    """
    Plot a bivariate normal distribution specified by BivariateNormal instance
    `bvn`, including contours of its PDF and regression lines (with conditional 
    standard deviations).

    Parameters
    ----------
    ax : mpl axes instance
        The axes to use for the plot

    bvn : BivariateNormal instance
        A bvn instance defining the BVN to plot

    rlines : boolean
        If True, plot the y|x and x|y regression lines

    regress : float
        If nonzero, vertical and horizontal lines are plotted illustrating
        regression to the mean for x = `regress * sig_x`

    samples : int
        If nonzero, `samples` samples from the BVN are plotted
    """
    x, y, xg, yg, xyg = bvn.xy_grid(100)
    log_pdf = log(bvn.pdf(xyg))
    log_pdf = log_pdf - log_pdf.max()  # values relative to the peak

    # Important to have the right aspect ratio!  Otherwise the symmetry axis
    # will appear incorrect.
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    aspect = bvn.sigs[0]/bvn.sigs[1]
    ax.set_aspect(aspect)

    # Contours of the log(PDF) for CL = 68.3%, 95.4%, 99.73%, 99.99%;
    # have also used natural steps in Q (-9, -4, -1):
    cont = ax.contour(xg, yg, log_pdf, [-9.667, -5.915, -3.090, -1.148], 
        colors=['gray', 'b', 'g', 'r'])
    clabel(cont, inline=1, fontsize=10)

    if rlines:
        y_x = bvn.y_x(x)
        ax.fill_between(x, y_x-bvn.csig_y, y_x+bvn.csig_y, facecolor='b', alpha=.2)
        ax.plot(x, y_x, 'b', lw=2, label='E$(y|x)$')
        x_y = bvn.x_y(y)
        ax.fill_betweenx(y, x_y-bvn.csig_x, x_y+bvn.csig_x, facecolor='g', alpha=.2)
        ax.plot(x_y, y, 'g', lw=2, label='E$(x|y)$')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if rlines:
        ax.legend(loc='upper left', fontsize=10)

    # PDF symmetry axis:
    ax_y = empty_like(x)
    for i, xx in enumerate(x):
        ax_y[i] = bvn.means[1] \
              + sign(bvn.rho)*bvn.sigs[1]*(xx - bvn.means[0])/bvn.sigs[0]

    ax.plot(x, ax_y, 'k--')

    # Illustrate regression to the mean:
    if regress:
        xl, xu = xlim()
        yl, yu = ylim()
        x = bvn.means[0] + regress*bvn.sigs[0]
        y = bvn.y_x(x)
        ax.plot([x, x, xl], [yl, y, y], 'c:', lw=2)

    # Plot some samples:
    if samples:
        xy = bvn.sample(samples)
        xvals, yvals = xy[:,0], xy[:,1]
        ax.scatter(xvals, yvals, c='k', linewidths=0, alpha=.4)


def plot_new_bvn(ax, rho, mu_x=0., mu_y=0., sig_x=1., sig_y=1., rlines=None,    
                 regress=None, samples=None):
    """
    Plot a bivariate normal distribution specified by its five parameters, 
    including contours of its PDF and regression lines (with conditional 
    standard deviations).

    Parameters
    ----------
    ax : mpl axes instance
        The axes to use for the plot

    rlines : boolean
        If True, plot the y|x and x|y regression lines

    regress : float
        If nonzero, vertical and horizontal lines are plotted illustrating
        regression to the mean for x = `regress * sig_x`

    samples : int
        If nonzero, `samples` samples from the BVN are plotted
    """
    means = array([mu_x, mu_y])
    sigs = array([sig_x, sig_y])
    bvn = BivariateNormal(means, sigs, rho)
    plot_bvn(ax, bvn, rlines, regress, samples)
