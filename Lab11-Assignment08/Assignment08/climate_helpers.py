"""
Tools for polynomial time series modeling of climate temperature data.

The climate data included here is the global annual mean surface air
temperature change data produced by NASA GISS, obtained from:

    Data.GISS: GISS Surface Temperature Analysis (GISTEMP)
    https://data.giss.nasa.gov/gistemp/

The data included here are from the "Graphs" link there:

    Data.GISS: GISS Surface Temperature Analysis: Analysis Graphs and Plots
    https://data.giss.nasa.gov/gistemp/graphs/

Specifically, the table included here is what is used for the "Global Annual
Mean Surface Air Temperature Change" plot there.

The methodology is explained in Hansen et al. (2010):

    Pubs.GISS: Hansen et al. 2010: Global surface temperature change
    https://pubs.giss.nasa.gov/abs/ha00510u.html

Specifically, GISS estimates *relative* temperature over land and sea
surfaces, from which it computes various spatial and temporal averages.
The temperature changes are reported relative to the time-averaged global
temperature from 1951-1980.  That average temperature, 14 deg C, may be
added to temperature change estimates to produce absolute temperatures.
However, the base temperature has an uncertainty of a few tenths of a
degree (C).  The relative temperature change is estimated much more
accurately.

Regarding the base period (from Hansen et al.):

    The GISS analysis uses 1951–1980 as the base period.
    The United States National Weather Service uses a 3 decade
    period to define “normal” or average temperature. When we
    began our global temperature analyses and comparisons with
    climate models, that climatology period was 1951–1980.

GISS updates its online estimates monthly.  The data here were obtained
on 2020-04-24.

NASA hosts a web site providing some of this data for the general public:

    Climate Change: Vital Signs of the Planet: Global Temperature
    https://climate.nasa.gov/vital-signs/global-temperature/

The anual global temperature data there, although sourced from GISS, differs
from the GISS data by +- 0.01 deg C for some years.  It may not use the
very latest GISS data.

2020-04-24  Revised for BDA20 by Tom Loredo
"""

import io

import numpy as np
import scipy
from numpy import *
from scipy import stats


__all__ = ['giss_temp_data', 'years', 'dTs',
           'MonomialBasis', 'ChebyshevBasis']


# empty 2018 entry omitted - "*" entries in both columns
giss_temp_data_raw = \
    """Land-Ocean Temperature Index (C)
--------------------------------

Year No_Smoothing  Lowess(5)
----------------------------
1880     -0.16     -0.09
1881     -0.08     -0.12
1882     -0.10     -0.16
1883     -0.16     -0.19
1884     -0.28     -0.23
1885     -0.32     -0.25
1886     -0.30     -0.26
1887     -0.35     -0.26
1888     -0.16     -0.26
1889     -0.10     -0.25
1890     -0.34     -0.24
1891     -0.22     -0.25
1892     -0.26     -0.26
1893     -0.31     -0.25
1894     -0.29     -0.23
1895     -0.21     -0.21
1896     -0.10     -0.19
1897     -0.10     -0.17
1898     -0.25     -0.15
1899     -0.16     -0.16
1900     -0.07     -0.19
1901     -0.15     -0.22
1902     -0.27     -0.25
1903     -0.36     -0.28
1904     -0.46     -0.31
1905     -0.26     -0.34
1906     -0.22     -0.36
1907     -0.39     -0.37
1908     -0.43     -0.39
1909     -0.49     -0.41
1910     -0.43     -0.41
1911     -0.44     -0.39
1912     -0.36     -0.35
1913     -0.34     -0.32
1914     -0.15     -0.31
1915     -0.14     -0.30
1916     -0.36     -0.30
1917     -0.46     -0.30
1918     -0.30     -0.30
1919     -0.28     -0.29
1920     -0.27     -0.28
1921     -0.19     -0.26
1922     -0.28     -0.25
1923     -0.26     -0.24
1924     -0.27     -0.23
1925     -0.22     -0.22
1926     -0.10     -0.22
1927     -0.22     -0.21
1928     -0.20     -0.20
1929     -0.36     -0.19
1930     -0.16     -0.19
1931     -0.09     -0.19
1932     -0.16     -0.18
1933     -0.28     -0.17
1934     -0.13     -0.16
1935     -0.20     -0.14
1936     -0.15     -0.11
1937     -0.03     -0.06
1938      0.00     -0.01
1939     -0.02      0.03
1940      0.13      0.06
1941      0.19      0.09
1942      0.07      0.11
1943      0.09      0.10
1944      0.20      0.07
1945      0.09      0.04
1946     -0.07      0.00
1947     -0.03     -0.04
1948     -0.11     -0.07
1949     -0.11     -0.08
1950     -0.17     -0.08
1951     -0.07     -0.07
1952      0.01     -0.07
1953      0.08     -0.07
1954     -0.13     -0.07
1955     -0.14     -0.06
1956     -0.19     -0.05
1957      0.05     -0.04
1958      0.06     -0.01
1959      0.03      0.02
1960     -0.02      0.03
1961      0.06      0.02
1962      0.04     -0.01
1963      0.05     -0.02
1964     -0.20     -0.04
1965     -0.11     -0.05
1966     -0.06     -0.06
1967     -0.02     -0.05
1968     -0.08     -0.03
1969      0.05     -0.02
1970      0.02     -0.01
1971     -0.08      0.00
1972      0.01      0.00
1973      0.16     -0.00
1974     -0.07      0.00
1975     -0.01      0.02
1976     -0.10      0.04
1977      0.18      0.07
1978      0.07      0.12
1979      0.16      0.16
1980      0.26      0.20
1981      0.32      0.21
1982      0.14      0.22
1983      0.31      0.21
1984      0.16      0.21
1985      0.12      0.22
1986      0.18      0.24
1987      0.32      0.27
1988      0.38      0.30
1989      0.27      0.33
1990      0.45      0.33
1991      0.40      0.32
1992      0.22      0.33
1993      0.23      0.33
1994      0.31      0.34
1995      0.45      0.37
1996      0.33      0.40
1997      0.46      0.42
1998      0.61      0.45
1999      0.39      0.47
2000      0.40      0.50
2001      0.54      0.53
2002      0.63      0.55
2003      0.62      0.59
2004      0.54      0.61
2005      0.68      0.62
2006      0.64      0.63
2007      0.66      0.63
2008      0.54      0.64
2009      0.66      0.64
2010      0.72      0.65
2011      0.61      0.66
2012      0.64      0.70
2013      0.68      0.74
2014      0.75      0.79
2015      0.90      0.83
2016      1.02      0.87
2017      0.92      0.91
2018      0.85      0.95
2019      0.98      0.98
"""

# Parse the data:
stream = io.StringIO(giss_temp_data_raw)
giss_temp_data = loadtxt(stream, skiprows=5, usecols=[0, 1])
years = giss_temp_data[:,0]
dTs = giss_temp_data[:,1]


class LinearBasis(object):
    """
    Base class for linear regression basis functions over a predictor
    scaled to [-1,1].
    """

    def __init__(self, deg, times, mean=0., sigma=1.):
        """
        Build a basis spanning the times in `times`, with degree
        `deg` (so `deg` + 1 terms).

        Parameters
        ----------

        deg : int
            Highest polynomial degree; `deg` should be 1 or greater

        times : float vector
            Times to evaluate the basis functions.

        `mean` and `sigma` define independent normal priors for coefficients
        on the basis; they may be scalars or length-`deg` arrays.

        Internally, time is mapped to x in [-1, 1].
        """
        self.deg = deg
        self.num = deg + 1
        self.times = times
        self.nt = len(self.times)
        self.durn = times[-1] - times[0]
        self.hdurn = 0.5*self.durn
        self.mid = times[0] + self.hdurn
        # Map the times to [-1, 1].
        self.xvals = (times - self.mid)/self.hdurn

        # Define the priors.
        self.means = zeros(self.num) + mean
        self.sigmas = ones(self.num)*sigma
        self.priors = [stats.norm(self.means[i], self.sigmas[i])
                       for i in range(self.num)]

    def func(self, betas):
        """
        Return the function defined by coefficients `beta` evaluated on the
        stored times.
        """
        return dot(self.basis, betas)

    def sample(self):
        """
        Return a random polynomial over the basis times, drawn from the prior.
        """
        coefs = array([self.priors[i].rvs() for i in range(self.num)])
        return dot(self.basis, coefs)


class MonomialBasis(LinearBasis):

    def __init__(self, deg, times, mean=0., sigma=1.):
        """
        Build a polynomial basis spanning the times in `times`, with degree
        `deg` (so `deg` + 1 terms), using simple monomials.

        Parameters
        ----------

        deg : int
            Highest polynomial degree; `deg` should be 1 or greater

        times : float vector
            Times to evaluate the basis functions.

        `mean` and `sigma` define independent normal priors for coefficients
        on the basis; they may be scalars or length-`deg` arrays.

        Internally, time is mapped to x in [-1, 1].
        """
        super().__init__(deg, times, mean, sigma)

        # Set the 1st basis directly.
        basis = [ones_like(times)]

        # Use recursion for the rest.
        for i in range(1, deg+1):
            new = self.xvals*basis[-1]
            basis.append(new)
        self.basis = array(basis).transpose()  # basis[t,d], row runs over time


class ChebyshevBasis(LinearBasis):

    def __init__(self, deg, times, mean=0., sigma=1.):
        """
        Build a Chebyshev polynomial basis spanning the times in `times`,
        with degree `deg` (so `deg` + 1 terms).

        `deg` should be 1 or greater.

        `times` is assumed to be in increasing order.

        `mean` and `sigma` define independent normal priors for coefficients
        on the basis; they may be scalars or length-`deg` arrays.

        This uses Chebyshev polynomials of the first kind; the first several
        are:

        T_0(x) = 1.
        T_1(x) = x
        T_2(x) = 2.*x**2 - 1.
        T_3(x) = 4.*x**3 - 3.*x
        T_4(x) = 8.*x**4 - 8.*x**2 + 1.
        T_5(x) = 16.*x**5 - 20.*x**3 + 5.*x

        These are provided as methods as a function of time; internally,
        time is mapped to x in [-1, 1].
        """
        super().__init__(deg, times, mean, sigma)

        # Set the 1st two (degree 0 & 1) directly.
        basis = [ones_like(times), self.xvals*ones_like(times)]

        # Use recursion for the rest.
        for i in range(2, deg+1):
            new = 2.*self.xvals*basis[-1] - basis[-2]
            basis.append(new)
        self.basis = array(basis).transpose()  # basis[t,d], row runs over time

    def T_0(self, t):
        t = asarray(t)
        return ones_like(t)

    def T_1(self, t):
        x = (asarray(t) - self.mid)/self.hdurn
        return x

    def T_2(self, t):
        x = (asarray(t) - self.mid)/self.hdurn
        return 2.*x**2 - 1.

    def T_3(self, t):
        x = (asarray(t) - self.mid)/self.hdurn
        return x*(4.*x**2 - 3.)

    def T_4(self, t):
        x = (asarray(t) - self.mid)/self.hdurn
        xx = x**2
        return 1. + xx*(8.*xx - 8.)

    def T_5(self, t):
        x = (asarray(t) - self.mid)/self.hdurn
        xx = x**2
        return x*(5. + xx*(-20. + 16.*xx))


def plot_temps():
    """
    Plot the temperature change time series data.
    """
    import matplotlib.pyplot as plt

    temp_fig = plt.figure(figsize=(8,5))
    plt.plot(years, dTs, 'b-', alpha=.3)
    plt.plot(years, dTs, ls='None', marker='o', ms=5, mfc='b', mew=0, alpha=.7)
    plt.xlabel('Year')
    plt.ylabel(r'$\Delta T$ (C)')
    return temp_fig


if __name__ == '__main__':
    from matplotlib.pyplot import *

    ion()

    plot_temps()
