"""
Timer context with audio alert, for use in Jupyter notebooks.
"""

import timeit

from IPython.display import Audio, display  # for audio alert


def done_alert():
    "Play an audio file as an alert."
    # From: https://forums.fast.ai/t/sound-alerts-in-jupyter-for-code-completion-and-exceptions/4614
    print('Task completed!')
    display(Audio(url='http://www.orangefreesounds.com/wp-content/uploads/2014/09/do-or-do-not-there-is-no-try.mp3', autoplay=True))


class Timer(object):
    """
    Context manager tracking elapsed time.
    """

    def __init__(self, name=None, units='auto', quiet=True):
        """
        Setup a timer context manager.

        If units=None, the elapsed time is not printed; it is available
        via `secs` and `msecs` attributes.

        If units='s' or 'ms', the appropriate attribute is printed when
        the context exits; if `name` is not None, it is used to label
        the output.  If units='auto', seconds are used for times >=1 s,
        otherwise milliseconds are used.
        """
        self.name = name
        self.units = units
        self.quiet = quiet

    def __enter__(self):
        if self.units:
            if self.name:
                print('* {}...'.format(self.name))
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.units == 'auto':
            if self.secs >= 1.:
                units = 's'
            else:
                units = 'ms'
        else:
            units = self.units
        if units == 's':
            print('-> Elapsed time = {:f} s'.format(self.secs))
        elif units == 'ms':
            print('-> Elapsed time = {:f} ms'.format(self.msecs))

        if not self.quiet:
            done_alert()
