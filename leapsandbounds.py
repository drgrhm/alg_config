#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import math
import numpy as np
import simulated_environment
import pickle
from util import format_runtime

R = 44  # Constant used for calculating b.
R2 = 32  # Constant used for the stopping condition (Appendix D, line 25).



def leaps_and_bounds(env, n, epsilon, delta, zeta, k0, theta_multiplier):
    """Implementation of LeapsAndBounds."""
    # This implementation makes some adjustments to the constants that control
    # how the failure probability budget zeta is allocated between the different
    # high-probability events that guarantee correctness.
    theta = k0 * 16. / 7
    k = 0
    while True:
        k += 1
        b = int(math.ceil(R * math.log(40 * 3 * n * k * (k + 1) / zeta) / (delta * epsilon * epsilon)))
        print('b={}, theta={}, total runtime so far={}'.format(b, theta, env.get_total_runtime()))
        q_hat = []
        for i in xrange(n):
            q_hat_i = ebgstop_slave_alg(env, i, b, delta, theta, k, epsilon, zeta, n)
            q_hat.append(q_hat_i)
        if np.min(q_hat) < theta:
            best_config_index = np.argmin(q_hat)
            return (best_config_index, q_hat[best_config_index], 4 * theta / (3 * delta))
        theta *= theta_multiplier


def ebgstop_slave_alg(env, i, b, delta, theta, k, epsilon, zeta, n):
    """Implementation of RuntimeEst with EBGStop, as in Appendix D in paper."""
    beta = 1.10
    t = b * theta  # Corresponds to T in the paper.
    tau = 4 * theta / (3 * delta)
    q = []
    sumq, sum_q_squared = 0, 0
    kk = 0  # Corresponds to l in the paper.

    for j in xrange(b):
        elapsed = 0
        if t > 0:
            _, elapsed, _ = env.run(config_id=i, instance_id=j, timeout=min(t, tau))

        t -= elapsed
        q.append(elapsed)
        sumq += elapsed
        sum_q_squared += elapsed * elapsed
        if t == 0:
            return theta

        q_mean = sumq / (j + 1)
        q_var = max((sum_q_squared - q_mean * sumq) / (j + 1), 0)

        if j + 1 > np.floor(np.power(beta, kk)):
            kk += 1
            alpha = np.floor(np.power(beta, kk)) / np.floor(np.power(beta, kk - 1))
            dk = (2.1 * k ** 1.5 * 2.61238 * (kk ** 1.1) * 10.5844 * n) / zeta
            x = alpha * np.log(3 * dk)

        if j > 0:
            confidence = np.sqrt(q_var * 2 * x / (j + 1)) + 3 * tau * x / (j + 1)
            lower_bound = q_mean - confidence
            if (1 + 3 * epsilon / 7) * lower_bound > theta and q_mean > theta:
                return theta
            d_prime = zeta / (40 * 3 * n * k * (k + 1) * j * (j + 1))
            r2 = math.ceil(-R2 * np.log(d_prime) / delta)
            # if j + 1 >= r2 and confidence <= epsilon / 3 * (q_mean + lower_bound):
            if j + 1 >= r2 and confidence <= (epsilon * q_mean) / (2 + 2 * epsilon):  # bug in original (as per CR paper)
                return q_mean
    return np.mean(q)



def main(epsilon, delta):
    parser = argparse.ArgumentParser(description='Executes LeapsAndBounds with a simulated environment.')
    parser.add_argument('--zeta', help='Zeta from the paper', type=float, default=0.1)
    parser.add_argument('--k0', help='Kappa_0 from the paper', type=float, default=1.)
    parser.add_argument('--theta-multiplier', help='Theta multiplier from the paper', type=float, default=1.25)
    parser.add_argument('--measurements-filename', help='Filename to load measurement results from', type=str, default='measurements.dump')
    parser.add_argument('--measurements-timeout', help='Timeout (seconds) used for the measurements', type=float, default=900.)
    args = vars(parser.parse_args())

    zeta = args['zeta']
    k0 = args['k0']
    theta_multiplier = args['theta_multiplier']
    results_file = args['measurements_filename']
    timeout = args['measurements_timeout']

    try: os.mkdir('results')
    except OSError: pass

    print("creating simulated environment")
    env = simulated_environment.Environment(results_file, timeout)
    num_configs = env.get_num_configs()

    print("running leaps_and_bounds")
    best_config_index, capped_avg, tau = leaps_and_bounds(env, num_configs, epsilon, delta, zeta, k0, theta_multiplier)
    print('best_config_index={}, capped_avg={}, tau={}'.format(best_config_index, capped_avg, tau))
    env.print_config_stats(best_config_index, tau=tau)
    print('total runtime: ' + format_runtime(env.get_total_runtime()))
    print('total resumed runtime: ' + format_runtime(env.get_total_resumed_runtime()))

    return best_config_index, env.get_total_runtime(), env.get_total_resumed_runtime()


if __name__ == '__main__':

    # deltas = [.95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05]
    deltas = [.5, .2]
    epsilons = [.1]

    results = []
    for epsilon in epsilons:
        for delta in deltas:
            print('----- epsilon={}, delta={} -----'.format(epsilon, delta))

            best_config_index, total_runtime, total_resumed_runtime = main(epsilon, delta)

            results.append({'best_config':best_config_index, 'epsilon':epsilon, 'delta':delta, 'total_runtime':total_runtime, 'total_resumed_runtime':total_resumed_runtime})

        with open(os.path.join('results', 'results_lb_eps=' + str(epsilon) + '.p'), 'wb') as f:
            pickle.dump(results, f)