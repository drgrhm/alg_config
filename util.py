from __future__ import division

import numpy as np
from math import log

day_in_seconds = 24. * 60. * 60.


def log2(x):
    return log(x,2)


def ecdf(vals, x, eps=1e-12):
    """
    Compute empirical cdf: P(X <= x) over the values vals
    """
    return np.sum(vals <= x, dtype=np.float32) / (np.shape(vals)[0] + eps)


def format_runtime(runtime):
    """ """
    return '{}s = {}m = {}h = {}d'.format(runtime, runtime / 60, runtime / 3600, runtime / (3600 * 24))