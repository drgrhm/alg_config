#!/usr/bin/python
#
# Copyright 2019 D R Graham


import argparse
import os
import numpy as np
import simulated_environment
from configuration_tester import ConfigurationTester
import pickle
from util import format_runtime, day_in_seconds
import time


def structured_procrastination_confidence(env, n, k0, theta_multiplier, total_time_budget, stop_times):
    """Implementation of Structured Procrastination with Confidence.
    todo:
    """

    configs = {}  # configurations
    for i in range(n):
        configs[i] = ConfigurationTester(i, k0, theta_multiplier)

    time_so_far = 0
    iter_count = 0
    t0, t1 = 0, 0

    results = []
    configs_r = []
    configs_total_time = []

    for stop_time in stop_times:
        while time_so_far < stop_time:

            i, _ = min([(cid, config.get_confidence_bound(iter_count)) for cid, config in configs.items()], key=lambda t: t[1])

            _, elapsed_time, lcb, instance_id = configs[i].execute_step(env, iter_count)
            time_so_far += elapsed_time

            if iter_count % 10000 == 0:
                t1 = time.time()
                print('iter_count={}, elapsed_time_since_last_print={:.0f}s, current_lcb={:3.2f}, fraction_of_time_so_far={:.5f}, current config_id={}, instance_count={}'.format(iter_count, t1 - t0, lcb, float(time_so_far) / float(total_time_budget), i, instance_id))
                t0 = time.time()

            iter_count += 1

        num_actives = [(i, c.get_num_active()) for i, c in configs.items()]
        i_star, i_star_r = max(num_actives, key=lambda t: t[1])
        print("------- cpu_days_so_far={}, iter_count={},  best_config_id={}, best_config_q={}, best_config_r={}, saving results -------".format(int(time_so_far / day_in_seconds), iter_count, i_star, configs[i_star].q, configs[i_star].r))

        results.append({'iterations':iter_count,
                        'best_config':i_star,
                        'best_config_theta':configs[i_star].theta,
                        'best_config_r':i_star_r,
                        'best_config_q':configs[i_star].q,
                        'total_runtime':time_so_far,
                        'total_resumed_runtime':env.get_total_resumed_runtime()})

        configs_r.append([(i, c.r) for i, c in configs.items()])
        configs_total_time.append([(i, env.get_runtime_per_config()[i]) for i, _ in configs.items()])

        with open(os.path.join('results', 'results_spc.p'), 'wb') as f:  # periodically save results
            pickle.dump(results, f)

        with open(os.path.join('results', 'configs_r_spc.p'), 'wb') as f:  # periodically save results
            pickle.dump(configs_r, f)

        with open(os.path.join('results', 'configs_total_time_spc.p'), 'wb') as f:  # periodically save results
            pickle.dump(configs_total_time, f)

    num_actives = [(i, c.get_num_active()) for i, c in configs.items()]
    i_star, _ = max(num_actives, key=lambda t:t[1])

    return i_star, configs


def main():
    parser = argparse.ArgumentParser(description='Executes Structured Procrastination with Confidence with a simulated environment.')
    parser.add_argument('--k0', help='Kappa_0 from the paper', type=float, default=1.)
    parser.add_argument('--theta-multiplier', help='Theta multiplier from the paper', type=float, default=2.)
    parser.add_argument('--measurements-filename', help='Filename to load measurement results from', type=str, default='measurements.dump')
    parser.add_argument('--measurements-timeout', help='Timeout (seconds) used for the measurements', type=float, default=900.)
    parser.add_argument('--total_time_budget', help='Total time (seconds) allowed', type=float, default=24.*60.*60.*2700.)  # 2700 CPU days;
    args = vars(parser.parse_args())

    k0 = args['k0']
    theta_multiplier = args['theta_multiplier']
    results_file = args['measurements_filename']
    timeout = args['measurements_timeout']
    total_time_budget = args['total_time_budget']

    try: os.mkdir('results')
    except OSError: pass

    print("creating simulated environment")
    env = simulated_environment.Environment(results_file, timeout)
    num_configs = env.get_num_configs()

    print("running structured_procrastination_confidence")

    step_size = int(day_in_seconds)  # CPU day, in second
<<<<<<< HEAD
    # stop_times = range(step_size, 10 * int(day_in_seconds), step_size) + range(10 * int(day_in_seconds), int(total_time_budget) + 1, 10 * step_size)  # check results at 1,2,3,..,9,10,20,30,... CPU days
    stop_times = range(step_size, 10 * int(day_in_seconds) + 1, step_size) + range(50 * int(day_in_seconds), int(total_time_budget) + 1, 50 * step_size)  # check results at 1,2,3,..,9,10,50,100,150,... CPU days
=======
    stop_times = list(range(step_size, 10 * int(day_in_seconds), step_size)) + list(range(10 * int(day_in_seconds), int(total_time_budget) + 1, 10 * step_size))  # check results at 1,2,3,..,9,10,20,30,... CPU days
>>>>>>> bb278d5166196274dc809e578117381d9d5fb355

    t0 = time.time()
    best_config_index, configs = structured_procrastination_confidence(env, num_configs, k0, theta_multiplier, total_time_budget, stop_times)
    t1 = time.time()

    print("")
    print("for total_time_budget={}".format(total_time_budget))
    print('best_config_index={}'.format(best_config_index))

    print("")
    print('total runtime: ' + format_runtime(env.get_total_runtime()))
    print('total resumed runtime: ' + format_runtime(env.get_total_resumed_runtime()))

    print("")
    print("Total real time to run: {}".format(t1 - t0))



if __name__ == '__main__':
    # np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

    main()
