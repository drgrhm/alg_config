import numpy as np
import csv
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import math


def proportion_optimal_solver(data_file, name, deltas):
    """For the solver/instance distribution <name>, plot the proportion of
        (epsilon, delta)-optimal configs agains epsilon, for various values of delta
    """
    df = pd.read_csv(data_file, sep=",", header=None)
    data = np.array(df.drop([0], axis=1))
    means = np.mean(data, axis=1)
    means = np.array(means)
    data = data[means != 0, :]
    means = np.mean(data, axis=1)
    opt = np.min(means)

    n_configs, n_instances = data.shape

    n_deltas = len(deltas)
    epsilons = {}
    for d, delta in enumerate(deltas):
        epsilons[d] = []
        for i in range(n_configs):
            runtimes = data[i, :]
            delta_quantile = int(n_instances - delta * n_instances)
            runtimes = np.sort(runtimes)
            theta = runtimes[delta_quantile]
            runtimes_cap = np.copy(runtimes)
            runtimes_cap[runtimes_cap > theta] = theta
            mean_cap = runtimes_cap.mean()
            epsilons[d].append((mean_cap / opt) - 1)
        epsilons[d].sort()

    matplotlib.rcParams.update({'font.size':16})
    f = plt.figure(figsize=(5 * n_deltas, 5))
    for d, delta in enumerate(deltas):

        n_eps = len(epsilons[d])
        eps_prop = (np.arange(n_eps) + 1) / n_eps

        ax = plt.subplot(1, n_deltas, d + 1)
        ax.plot(epsilons[d], eps_prop, linewidth=6.0, color='#12B7EC')
        ax.set_xscale("log")
        plt.xlim((.01, 10e7))
        plt.title(r'{}: $\delta$ = {:.3f}'.format(name, delta))
        plt.xlabel(r'$\epsilon$')
        if d == 0:
            plt.ylabel('Proportion of configurations')

        f.tight_layout()

    plt.savefig(os.path.join("img", "prop_optimal_{:s}.pdf".format(name)), bbox_inches='tight')


def proportion_optimal_deltas(inputs, delta):
    """For various delta and epsilon, plot proportion of configs that are (epsilon, delta)-optimal
        for each config in <inputs>
    """
    names = ["-".join(input.split('-')[1:3]) + r': $\delta$ = ' + str(delta) for input in inputs]

    matplotlib.rcParams.update({'font.size':16})
    f = plt.figure(figsize=(5 * 3, 5))

    for x, input in enumerate(inputs):
        data_file = os.path.join('data', input)
        df = pd.read_csv(data_file, sep=",", header=None)
        cols = ['runtime_{:d}'.format(i) for i in range(1000)]
        cols = ['config'] + cols
        df.columns = cols
        data = np.array(df.drop(['config'], axis=1))
        means = np.mean(data, axis=1)
        data = data[means != 0, :]
        means = np.mean(data, axis=1)
        opt = np.min(means)
        n_configs, n_instances = data.shape

        epsilons = []
        for i in range(n_configs):
            runtimes = data[i, :]
            runtimes = [min(rt, 300) for rt in runtimes]
            runtimes = [max(rt, .001) for rt in runtimes]
            runtimes = np.sort(runtimes)
            delta_quantile = int(n_instances - delta * n_instances)
            theta = runtimes[delta_quantile]
            runtimes_cap = np.copy(runtimes)
            runtimes_cap[runtimes_cap > theta] = theta
            mean_cap = runtimes_cap.mean()
            epsilons.append((mean_cap / opt) - 1)
        epsilons.sort()
        n_eps = len(epsilons)
        eps_prop = (np.arange(n_eps) + 1) / n_eps
        ax = plt.subplot(1, 3, x + 1)
        ax.plot(epsilons, eps_prop, linewidth=6.0, color='#12B7EC')
        ax.set_xscale("log")
        plt.title(names[x])
        plt.xlabel(r'$\epsilon$')
        if x == 0:
            plt.ylabel("Proportion of configurations")
        f.tight_layout()

    plt.savefig(os.path.join("img", "proportion_optimal_deltas.pdf"), bbox_inches='tight')


if __name__ == "__main__":

    try:
        os.mkdir("img")
    except OSError:
        pass

    data_file = os.path.join('data', '1000samples-SPEAR-SWV-all604inst-results.txt')
    proportion_optimal_solver(data_file, 'SPEAR-SWV', deltas=[.001, .01, .1, .5])

    input_files = ['1000samples-SPEAR-IBM-all765inst-results.txt',
                   '1000samples-CPLEX-BIGMIX-all1510inst-results.txt',
                   '1000samples-CPLEX-CORLAT-REG-results.txt']
    proportion_optimal_deltas(input_files, delta=.001)






