from __future__ import division

import os
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util import day_in_seconds
import simulated_environment
import numpy as np


def plot_results():

    parser = argparse.ArgumentParser(description='Plots the saved results from the configuration procedures.')
    parser.add_argument('--measurements-timeout', help='Timeout (seconds) used for the measurements', type=float, default=900.)
    args = vars(parser.parse_args())

    timeout = args['measurements_timeout']

    try:
        with open(os.path.join('results', 'results_lb_grid.p'), 'rb') as f:
            results_lb = pickle.load(f)
    except IOError as err:
        print(err, "no lb results saved")
        return

    try:
        with open(os.path.join('results', 'results_sp_grid.p'), 'rb') as f:
            results_sp = pickle.load(f)
    except IOError as err:
        print(err, "no sp results saved")
        return

    try:
        with open(os.path.join('results', 'results_spc.p'), 'rb') as f:
            results_spc = pickle.load(f)
    except IOError as err:
        print(err, "no spc results saved")
        return

    try:
        with open(os.path.join('results', 'means_uncap.p'), 'rb') as f:
            means_uncap = pickle.load(f)
    except IOError:
        env = simulated_environment.Environment('measurements.dump', timeout)
        results = env.get_results()
        results_uncap = [[timeout if x > timeout else x for x in res] for res in results]
        means_uncap = [np.mean(res) for res in results_uncap]
        with open(os.path.join('results', 'means_uncap.p'), 'wb') as f:
            pickle.dump(means_uncap, f)

    best_config_means_lb = [means_uncap[res['best_config']] for res in results_lb]
    times_lb = [res['total_runtime'] / day_in_seconds for res in results_lb]

    best_config_means_sp = [means_uncap[res['best_config']] for res in results_sp]
    times_sp = [res['total_runtime'] / day_in_seconds for res in results_sp]

    best_config_means_spc = [means_uncap[res['best_config']] for res in results_spc]
    times_spc = [res['total_runtime'] / day_in_seconds for res in results_spc]

    colours_lb = [5. * config['delta'] ** 4 for config in results_lb]
    sizes_lb = [150. * config['epsilon'] for config in results_lb]

    colours_sp = [5. * config['best_config_delta'] ** 4 for config in results_sp]
    sizes_sp = [150. * config['epsilon'] for config in results_sp]

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    plt.rcParams.update({'font.size': 23})

    ax1.plot(times_spc, best_config_means_spc, '.-', color='mediumblue', alpha=.5, label="SPC", linewidth=2, markersize=5)
    ax1.scatter(times_lb, best_config_means_lb, color='maroon', alpha=.5, c=colours_lb, cmap=plt.cm.get_cmap('Reds'), s=sizes_lb, label="LB")
    ax1.scatter(times_sp, best_config_means_sp, color='green', alpha=.7, c=colours_sp, cmap=plt.cm.get_cmap('Greens'), s=sizes_sp, label="SP")

    opt = min(means_uncap)
    del_max = ax1.scatter(-1000, opt, color='lightcoral', alpha=.8)  # dummy points off plot to force legend style
    del_min = ax1.scatter(-1000, opt, color='maroon', alpha=.1)
    eps_min = ax1.scatter(-1000, opt, color='maroon', alpha=.4, s=np.min([2. * (1. / config['epsilon'])**2 for config in results_lb]))
    eps_max = ax1.scatter(-1000, opt, color='maroon', alpha=.4, s=np.max([1.5 * (1. / config['epsilon'])**2 for config in results_lb]))
    ax1.axhline(opt, linestyle='--', color='dodgerblue', label="OPT", linewidth=1)
    # ax1.set_title("Mean runtime of sol'n returned \n(capped at max cap)")
    ax1.set_xlabel("CPU days")
    ax1.set_ylabel("Mean runtime")
    plt.xscale('log')
    # plt.xticks([1] + [300 * i for i in range(1, 9)])
    ax1.set_xlim(0, 3500)
    ax1.set_ylim(19, 21.7)
    ax1.locator_params(axis='y', nbins=8)

    handles, labels = ax1.get_legend_handles_labels()
    handles[1], handles[2], handles[3] = handles[2], handles[3], handles[1]
    labels[1], labels[2], labels[3] = labels[2], labels[3], labels[1]
    legend1 = plt.legend(handles, labels, loc=(.724, .751), prop={'size':11.6})
    plt.gca().add_artist(legend1)
    legend2 = plt.legend([eps_min, eps_max, del_min, del_max], [r"$\epsilon={}$".format(0.1), r"$\epsilon={}$".format(0.9), r"$\delta={}$".format(0.1), r"$\delta={}$".format(0.5),], loc=(.846,.7515), prop={'size': 11.5})

    rect = patches.Rectangle((346, 21), 2750, 0.63, linewidth=1 ,facecolor='none')
    ax1.add_patch(rect)

    plt.savefig('img/mean_plot.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    try: os.mkdir('img')
    except OSError: pass

    plot_results()

