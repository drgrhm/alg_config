#!/usr/bin/python
#
# Copyright 2018 Google LLC, 2019 D R Graham
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

import os
import argparse
import heapq
import pickle
import math
import numpy as np
import simulated_environment
from util import format_runtime, day_in_seconds

C = 12.  # constant for l_i


def structured_procrastination(env, n, epsilon, zeta, k0, k_bar, theta_multiplier, stop_times, deltas):
    """Implementation of Structured Procrastination."""
    # The names of the variables used here agree with the pseudocode in the paper,
    # except q is used instead of the paper's upper-case Q, and qq is used instead
    # of the paper's lower-case q. The pseudocode overloads l: here, ll is used to
    # represent the scalar (appears as l in paper), and l is used to represent the
    # array (appears as l_i in the paper). For efficiency, we implement the argmin
    # in line 10 of the paper with a heap.
    k, l, q, qq, r, r_sum, heap = [], [], [], [], [], [], []
    beta = math.log(k_bar / k0, 2)
    for i in range(n):  # Line 2 in paper.
        k.append(0)
        l.append(int(math.ceil(C / (epsilon * epsilon) * math.log(3 * beta * n / zeta))))
        q.append([])
        qq.append(0)
        r.append([])
        r_sum.append(0)
        heapq.heappush(heap, (0, i))
        for ll in range(l[i]):  # Line 6 in paper.
            r[i].append(0)
            q[i].append((ll, k0))

    # Main loop.
    results = []
    configs_r = []
    configs_total_time = []

    current_delta = 1
    iter_count = 0
    time_so_far = 0.
    for delta in deltas:
        while current_delta > delta:  # Line 9, but stop when target delta reached.
    # for stop_time in stop_times:
    #     while time_so_far < stop_time:
            iter_count += 1
            _, i = heapq.heappop(heap)
            ll, theta = q[i].pop(0)
            if r[i][ll] == 0:  # Line 12 in paper.
                k[i] += 1

                qq[i] = int(math.ceil(C / (epsilon * epsilon) * math.log(3 * beta * n * k[i] * k[i] / zeta)))

            did_timeout, elapsed, _ = env.run(config_id=i, timeout=theta, instance_id=ll)
            if not did_timeout:  # Line 15 in paper.
                r_sum[i] += elapsed - r[i][ll]
                r[i][ll] = elapsed
            else:
                r_sum[i] += theta - r[i][ll]
                r[i][ll] = theta
                q[i].append((ll, theta_multiplier * theta))
            while len(q[i]) < qq[i]:  # Line 20 in paper.
                l[i] += 1
                r[i].append(0)
                q[i].insert(0, (l[i] - 1, theta))
            time_so_far += elapsed
            heapq.heappush(heap, (r_sum[i] / k[i], i))  # Bookeeping for the heap.

            i_star = np.argmax(r_sum)
            current_delta = math.sqrt(1 + epsilon) * qq[i_star] / k[i_star]
        print("------- cpu_days_so_far={}, best_config_id={}, delta={}, theta={}, q={}, k={}. saving results -------".format(int(time_so_far / day_in_seconds), i_star, current_delta, theta, qq[i_star], k[i_star]))

        results.append({'iterations':iter_count,
                        'best_config':i_star,
                        'best_config_theta':theta,
                        'best_config_delta':current_delta,
                        'best_config_q':qq[i_star],
                        'best_config_k':k[i_star],
                        'total_runtime':time_so_far,
                        'total_resumed_runtime':env.get_total_resumed_runtime()})

        configs_r.append([(i, k[i]) for i in range(n)])
        configs_total_time.append([(i, env.get_runtime_per_config()[i]) for i in range(n)])

        with open(os.path.join('results', 'results_sp_eps=' + str(epsilon) + '.p'), 'wb') as f:  # periodically save results
            pickle.dump(results, f)

        with open(os.path.join('results', 'configs_r_sp.p'), 'wb') as f:  # periodically save results
            pickle.dump(configs_r, f)

        with open(os.path.join('results', 'configs_total_time_sp.p'), 'wb') as f:  # periodically save results
            pickle.dump(configs_total_time, f)

    return i_star, current_delta


def main(epsilon, deltas):
    parser = argparse.ArgumentParser(description='Executes Structured Procrastination with a simulated environment.')
    parser.add_argument('--epsilon', help='Epsilon from the paper', type=float, default=0.1)
    # parser.add_argument('--delta', help='Delta from the paper', type=float, default=0.2)
    parser.add_argument('--zeta', help='Zeta from the paper', type=float, default=0.1)
    parser.add_argument('--k0', help='Kappa_0 from the paper', type=float, default=1.)
    parser.add_argument('--k-bar', help='bar{Kappa} from the paper', type=float, default=1000000.)
    parser.add_argument('--theta-multiplier', help='Theta multiplier from the paper', type=float, default=2.0)
    parser.add_argument('--measurements-filename', help='Filename to load measurement results from', type=str, default='measurements.dump')
    parser.add_argument('--measurements-timeout', help='Timeout (seconds) used for the measurements', type=float, default=900.)
    parser.add_argument('--total-time-budget', help='Total time (seconds) allowed', type=float, default=2160000000.)  # 86400 seconds = 1 CPU day; 103680000 == 1200 CPU days
    args = vars(parser.parse_args())

    # epsilon = args['epsilon']
    # delta = args['delta']
    zeta = args['zeta']
    k0 = args['k0']
    k_bar = args['k_bar']
    theta_multiplier = args['theta_multiplier']
    results_file = args['measurements_filename']
    timeout = args['measurements_timeout']
    total_time_budget = args['total_time_budget']

    try: os.mkdir('results')
    except OSError: pass

    print("creating simulated environment")
    env = simulated_environment.Environment(results_file, timeout)
    num_configs = env.get_num_configs()

    print("running structured_procrastination")

    step_size = int(day_in_seconds)  # CPU day, in second
    stop_times = list(range(step_size, 10 * int(day_in_seconds), step_size)) + list(range(10 * int(day_in_seconds), int(total_time_budget) + 1, 10 * step_size))  # check results at 1,2,3,..,9,10,20,30,... CPU days

    best_config_index, delta = structured_procrastination(env, num_configs, epsilon, zeta, k0, k_bar, theta_multiplier, stop_times, deltas)

    print('best_config_index={}, delta={}'.format(best_config_index, delta))
    env.print_config_stats(best_config_index)

    print('total runtime: ' + format_runtime(env.get_total_runtime()))
    print('total resumed runtime: ' + format_runtime(env.get_total_resumed_runtime()))


if __name__ == '__main__':
    epsilons = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
    deltas = [.5, .4, .3, .2, .1]
    results = []

    for epsilon in epsilons:

        print("running spc with epsilon={} for deltas={}".format(epsilon, deltas))
        main(epsilon, deltas)

        with open(os.path.join('results', 'results_sp_eps={}.p'.format(epsilon)), 'rb') as f:
            results_eps = pickle.load(f)

            for res in results_eps:
                res['epsilon'] = epsilon
                results.append(res)

    with open(os.path.join('results', 'results_sp_grid.p'), 'wb') as f:
        pickle.dump(results, f)

