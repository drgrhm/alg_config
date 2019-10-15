from __future__ import division

import os
import math
import argparse
import pickle
import matplotlib.pyplot as plt
from util import day_in_seconds


def plot_results(epsilon):

    # parser = argparse.ArgumentParser(description='Plots the saved results from the configuration procedures.')
    # parser.add_argument('--epsilon', help='Epsilon used to run procedures', type=float, default=0.1)
    # args = vars(parser.parse_args())
    #
    # epsilon = args['epsilon']

    try:
        with open(os.path.join('results', 'results_lb_eps=' + str(epsilon) + '.p'), 'rb') as f:
            results_lb = pickle.load(f)
    except IOError as err:
        print(err, "no lb results saved for epsilon=".format(epsilon))
        return

    try:
        with open(os.path.join('results', 'results_sp_eps=' + str(epsilon) + '.p'), 'rb') as f:
            results_sp = pickle.load(f)
    except IOError as err:
        print(err, "no sp results saved for epsilon=".format(epsilon))
        return

    try:
        with open(os.path.join('results', 'results_spc.p'), 'rb') as f:
            results_spc = pickle.load(f)
    except IOError as err:
        print(err, "no spc results saved")
        return

    deltas_lb = [res['delta'] for res in results_lb]
    times_lb = [res['total_runtime'] / day_in_seconds for res in results_lb]

    deltas_sp = [res['best_config_delta'] for res in results_sp]
    times_sp = [res['total_runtime'] / day_in_seconds for res in results_sp]

    deltas_spc = [math.sqrt(1 + epsilon) * res['best_config_q'] / res['best_config_r'] for res in results_spc]  # todo: not correct delta!
    times_spc = [res['total_runtime'] / day_in_seconds for res in results_spc]

    ###
    deltas_sp = deltas_sp[2:]
    times_sp = times_sp[2:]
    ###

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xscale('log')

    ax.plot(times_lb, deltas_lb, '.', color='blueviolet', linewidth=1.6, markersize=7)
    ax.plot(times_sp, deltas_sp, '.-', color='lightcoral', linewidth=1.6)
    ax.plot(times_spc, deltas_spc, '.-', color='dodgerblue', linewidth=1.6)

    ax.set_ylabel(r'$\delta$', fontsize=18)
    ax.set_xlabel(r"Time to find $(\epsilon, \delta)$-optimal" + " solution (CPU days) \n ($\epsilon = 0.05$)", fontsize=14)

    # ax.axhline(y=0.2, linestyle='--', color='#BBBBBB')
    # ax.legend(['LeapsAndBounds', 'Structured Procrastination', 'Structured Procrastination\nwith Confidence'], frameon=False, loc=(.63, .84), fontsize=12)
    # # ax.set_axis_bgcolor('#EFEFEF')
    # # ax.grid(which='major', axis='y', linestyle='-', color='#CCCCCC')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.set_xlim(0, 5*10**4)

    plt.savefig(os.path.join('img', 'delta_plot_eps=' + str(epsilon) + '.pdf'), bbox_inches='tight')


if __name__ == '__main__':

    try: os.mkdir('img')
    except OSError: pass

    plot_results(epsilon=.1)