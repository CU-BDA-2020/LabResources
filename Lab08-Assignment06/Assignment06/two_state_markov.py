"""
A class for exploring the behavior of 2-state Markov chains.

Created Mar 20, 2015 by Tom Loredo
2020-03-10:  Revised for BDA20; added dither option
"""

from matplotlib.pyplot import *
from numpy import *
from scipy import stats


__all__ = ['TwoStateMarkovChain']


def sig_alpha(n, ntot):
    """
    Posterior standard deviation for a binomial probability inferred from
    n successes out of ntot trials, using a flat prior.

    This uses the formula for the standard deviation of a gamma dist'n.
    """
    return sqrt((n+1.)*(ntot-n+1.) / (ntot+2.)**2 / (ntot+3.))


class TwoStateMarkovChain:
    """
    A class for exploring the behavior of a 2-state Markov chain.
    """

    def __init__(self, alpha, beta, states=(0, 1), l=None):
        """
        Define a 2-state Markov chain from its state-change probabilities.

        Parameters
        ----------

        alpha, beta : float
            Probabilities for changing state; `alpha` is the probability for
            changing from the first state to the second, `beta` is the
            probability for changing from the second to the first.

        states : 2-tuple
            Labels for the two states; default is to label the first state
            with the integer 0, and the second with 1.  The labels should
            be legitimate dict keys.

        l : int
            Length of simulated sample paths; can also be separately set or
            changed with set_path_length().
        """
        self.alpha = alpha
        self.beta = beta
        self.state_a, self.state_b = states  # labels for states

        # Map state labels to 0, 1:
        self.s2int = {self.state_a: 0, self.state_b: 1}
        # Map 0, 1 to state labels:
        self.int2s = {0: self.state_a, 1: self.state_b}

        # Transition matrix; trans[j,i] = prob for move from i to j;
        # this is the right-multiplying form.
        self.trans = array([[1.-alpha, beta], [alpha, 1.-beta]])

        # Transition samplers from states 0, 1; note that the binom param is
        # the "success" (state=1) probability:
        self.samplers = [stats.binom(1, self.trans[1,0]).rvs,
                         stats.binom(1, self.trans[1,1]).rvs]

        # Equillibrium dist'n:
        self.p0_eq = beta/(alpha + beta)
        self.p1_eq = alpha/(alpha + beta)

        if l is not None:
            self.set_path_length(l)
        else:
            self.length = None

    def set_path_length(self, l):
        """
        Set the length of simulated sample paths, in terms of the # of
        steps (i.e., not counting the initial sample).  The stored paths
        will have l+1 entries (including the initial sample).
        """
        self.length = l
        self.times = arange(l+1)
        self.paths = []

    def sim_path(self, sampler, by_label=True):
        """
        Simulate a single sample path, appending it to `self.paths` as an
        array of states.

        Parameters
        ----------

        sampler : function
            A function that takes no arguments that returns a state sampled
            from the initial state distribution.

        by_label : bool
            Indicates whether sampler() returns a state label (if `True`) or
            an integer in {0,1} corresponding to the first & second states.
        """
        if self.length is None:
            raise ValueError('Set path length before simulating!')
        # path will store integers (0,1) indicating the state.
        path = empty(self.length+1, dtype=int)
        if by_label:
            path[0] = self.s2int[sampler()]
        else:
            path[0] = sampler()
        for i in range(1, self.length+1):
            path[i] = self.samplers[path[i-1]]()
        self.paths.append(path)

    def sim_paths(self, n, sampler, l=None, by_label=True):
        """
        Simulate new sample paths.  If the path length `l` is not provided,
        the current value is used and the new paths are appended to the
        current path list.  If `l` is provided, a new path list is created,
        with paths of length `l`.

        Parameters
        ----------

        n : int
            Number of sample paths to simulate

        sampler : function
            A function that takes no arguments that returns a state sampled
            from the initial state distribution.

        l : int
            Length of sample paths; specify if a new set of paths is sought

        by_label : bool
            Indicates whether sampler() returns a state label (if `True`) or
            an integer in {0,1} corresponding to the first & second states.
        """
        if l:
            self.set_path_length(l)
        for i in range(n):
            self.sim_path(sampler, by_label)

    def plot_evol(self, np=5, alpha_trace=.02, dither=0., figsize=(15,6)):
        """
        Make a plot depicting the evolution of the Markov process, displaying
        a trace plot with simulated sample paths, and panels using histograms
        of states at various times to show how the marginal distribution is
        evolving.

        Parameters
        ----------

        np : int
            Number of panels showing marginal PMF at time slices

        alpha_trace : float
            Opacity for lines and markers in the trace plot

        dither : float
            Half-width of vertical dither spanned by the traces, to aid
            visibility
        """
        fig = figure(figsize=figsize)

        # Plot a top panel with the trace plot.
        subplot(211)
        # Dither spacing for the y coordinate, to aid visibility.
        dd = 2.*dither/(len(self.paths)-1)
        d = dither
        for path in self.paths:
            # Show sample paths as squares connected by lines.
            plot(self.times, path+d, 'b-', alpha=alpha_trace)
            plot(self.times, path+d, 'gs', alpha=alpha_trace)
            d -= dd
        xlabel('')  # no xlabel, to avoid overlap with lower panels
        ylabel('State')
        ylim(-.2, 1.2)
        yticks([0,1])

        # Panels with time slices:

        # First panel, for t=0 (init step), shows y labels.
        axes = subplot(2,np,np+1)
        self.marg_pmf(axes, 0, show_yticks=True)
        # Middle panels at appropriate intermediate times:
        for i in range(1, np-1):
            axes = subplot(2,np,np+i+1)
            self.marg_pmf(axes, i*self.length//(np-1))  # note integer division
        # Last panel always last time:
        axes = subplot(2,np,2*np)
        self.marg_pmf(axes, -1)

        return fig

    def marg_pmf(self, axes, t, show_target=True, show_yticks=False):
        """
        Plot PMF estimate based on samples from the marginal dist'n at time t
        on the provided axes.
        """
        paths = asarray(self.paths)
        ntot = len(paths[:,t])
        n1 = sum(paths[:,t])
        n0 = ntot - n1
        p0, p1 = 1.*n0/ntot, 1.*n1/ntot
        sig0 = sig_alpha(n0, ntot)
        sig1 = sig_alpha(n1, ntot)
        w = .5  # bar width
        # axes.bar([-w/2, 1.-w/2], [p0, p1], w, alpha=.5)
        axes.bar([0, 1], [p0, p1], w, alpha=.5)
        axes.errorbar([0, 1], [p0, p1], yerr=[sig0, sig1],
                      ecolor='k', lw=2, capsize=8, ls='None')
        if show_target:
            # Dashed lines to show target PMF:
            tp0, tp1 = self.p0_eq, self.p1_eq
            axes.plot([-w, .5, .5, 1+w], [tp0, tp0, tp1, tp1], 'g-', lw=2, alpha=.5)
        axes.set_xlim(-w, 1+w)
        axes.set_xticks([0,1])
        axes.set_xlabel('State')
        axes.set_ylim(ymin=0, ymax=1.2)
        if show_yticks:
            axes.set_yticks([0, .5, 1.])
            axes.set_ylabel('PMF')
        else:
            axes.set_yticks([])
        # Show time:
        if t == -1:
            ttxt = self.length
        else:
            ttxt = t
        axes.text(.8, 1.05, '$t={}$'.format(ttxt))
