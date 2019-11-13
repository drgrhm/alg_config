#
# Copyright 2019 D R Graham

from collections import deque
from math import ceil, floor, log, sqrt
import numpy as np
import bisect


class ConfigurationTester():
    """

    """

    def __init__(self, cid, k0, theta_multiplier, update_lcb_every=1000):
        """
        Parameters:
            cid: config id
            k0 : kappa 0
            theta_multiplier : how much to increase theta by after timing out
            update_lcb_every : only re-compute the lcb for this config if at least this many iterations have elapsed (for efficiency)
        """

        self.cid = cid
        self.theta = k0  # current runtime cap, initially the initial value
        self.r = 0  # number of active instances
        self.q = 1.
        self.Q = deque()  # double ended queue
        self.theta_multiplier = theta_multiplier

        self.instance_runtimes_capped = {}  # mapping from instances to most recent capped runtime
        self.unique_values = []  # sorted list of unique runtime values seen
        self.unique_value_counts = {}

        self.total_time = 0  # time spent runing this configuration

        self.lcb = 0
        self.update_lcb_every = update_lcb_every
        self.t_last_update_lcb = -1


    def execute_step(self, env, t):
        """
        Execute one step of the algorithm for this configuration.
        """

        if len(self.Q) < self.q:
            self.r = self.r + 1
            l = self.r
        else:
            l, theta = self.Q.pop()
            self.theta = theta

        did_timeout, elapsed, resumed_elapsed = env.run(config_id=self.cid, timeout=self.theta, instance_id=l)  # get the runtime of config <cid> in instance <l>

        if did_timeout:
            rt = self.theta
            self.Q.appendleft((l, self.theta_multiplier * self.theta))
        else:
            rt = elapsed

        rt = np.round(rt, decimals=3)  # round to milliseconds

        self.total_time += rt

        self._update_runtime_values(l, rt)

        self.lcb = self._compute_confidence_bound(t)
        self.t_last_update_lcb = t

        if self.r <= 1 or t <= 1:
            self.q = 25.
        else:
            self.q = ceil(25. * log(t * log(self.r, 2), 2))

        return did_timeout, elapsed, self.lcb, len(self.instance_runtimes_capped)


    def _compute_confidence_bound(self, t):
        """
        Compute the lcb from the paper.
        """
        n = len(self.instance_runtimes_capped)
        ecdf = []  # empirical cumulative distribution of runtime values
        lcb = 0.
        for i in range(len(self.unique_values)):
            rt = self.unique_values[i]
            if i == 0:
                rt_low = 0.
                ecdf.append(self.unique_value_counts[rt] / n)
                g = 0.
            else:
                rt_low = self.unique_values[i - 1]
                ecdf.append(ecdf[i - 1] + self.unique_value_counts[rt] / n)
                g = ecdf[i - 1]

            lcb += (rt - rt_low) * self.beta(1. - g, t)

        return lcb


    def get_confidence_bound(self, t):
        """
        Returns the lcb for this config, re-computing it as necessary.
        """
        if t - self.t_last_update_lcb > self.update_lcb_every:  # re-compute lcb if enough iterations have elapsed
            self.lcb = self._compute_confidence_bound(t)
            self.t_last_update_lcb = t

        return self.lcb


    def beta(self, p, t):
        """
        Implementation of Beta function from paper.
        """
        k = floor(log(1 / p, 2))

        if self.r == 0:  # setting initial values so that we don't take logs of 0
            _r = 1.
        else:
            _r = self.r
        if t == 0:
            _t = 1
        else:
            _t = t
        if k == 0:
            eps = 3. / _r
        else:
            eps = sqrt(9 * 2 ** k * log(k * _t) / _r)

        if eps <= 0.5:
            return p / (1 + eps)
        else:
            return 0.


    def get_num_active(self):
        """
        Returns the number of active instances for this config.
        """
        return self.r


    def _update_runtime_values(self, instance_id, new_rt, eps=1e-6):
        """
        Maintains the sorted list of unique runtime values and their counts.
        """

        if instance_id in self.instance_runtimes_capped:
            old_rt = self.instance_runtimes_capped[instance_id]
            rt_count = self.unique_value_counts[old_rt]
            if rt_count > 1:
                self.unique_value_counts[old_rt] -= 1
            else:  # remove rt from unique values
                delete_ind = bisect.bisect_left(self.unique_values, old_rt)
                del self.unique_values[delete_ind]
                del self.unique_value_counts[old_rt]

        insert_ind = bisect.bisect_left(self.unique_values, new_rt)  # get new insert index
        n = len(self.unique_values)

        if insert_ind < n:  # insert a new value
            if abs(self.unique_values[insert_ind] - new_rt) > eps:  # sufficiently different from other unique values
                self.instance_runtimes_capped[instance_id] = new_rt  # save/update runtime value for this instance with new theta
                bisect.insort(self.unique_values, new_rt)
                self.unique_value_counts[new_rt] = 1
            else:
                self.instance_runtimes_capped[instance_id] = self.unique_values[insert_ind]
                self.unique_value_counts[self.unique_values[insert_ind]] += 1
        else:  # new_value is greater than all existing unique_values
            if n == 0 or abs(self.unique_values[n - 1] - new_rt) > eps:  # first value or sufficiently different from largest unique value
                self.instance_runtimes_capped[instance_id] = new_rt
                self.unique_values.append(new_rt)
                self.unique_value_counts[new_rt] = 1
            else:
                self.instance_runtimes_capped[instance_id] = self.unique_values[n - 1]
                self.unique_value_counts[self.unique_values[n - 1]] += 1

