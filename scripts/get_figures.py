import gflags
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

from get_result_data import get_data_from_single_file

gflags.DEFINE_string('log_folder', '../log', 'log folder')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_nodes(scene, nodes):
    rrt_nodes = nodes['rrt']
    astar_nodes = nodes['astar']

    plt.figure()

    bpl = plt.boxplot(rrt_nodes, positions=np.array(range(len(rrt_nodes))) * 2.0 - 0.4, widths=0.5)
    bpr = plt.boxplot(astar_nodes, positions=np.array(range(len(astar_nodes))) * 2.0 + 0.4, widths=0.5)

    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    plt.plot([], c='#D7191C', label='RRT')
    plt.plot([], c='#2C7BB6', label='Hybrid A*')

    ticks = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '0.05', '0.1', '0.25', '0.5']

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)

    plt.xlabel(r'$p_{max}$')
    plt.ylabel('N. Explored Nodes')

    if scene == 'scene_1':
        plt.legend(loc='upper left')

    plt.savefig('../figs/' + scene + '_nodes.png', bbox_inches='tight')


def plot_checks(scene, checks):
    rrt_checks = checks['rrt']
    astar_checks = checks['astar']

    plt.figure()

    bpl = plt.boxplot(rrt_checks, positions=np.array(range(len(rrt_checks))) * 2.0 - 0.4, widths=0.5)
    bpr = plt.boxplot(astar_checks, positions=np.array(range(len(astar_checks))) * 2.0 + 0.4, widths=0.5)

    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    plt.plot([], c='#D7191C', label='RRT')
    plt.plot([], c='#2C7BB6', label='Hybrid A*')

    ticks = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '0.05', '0.1', '0.25', '0.5']

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)

    plt.xlabel(r'$p_{max}$')
    plt.ylabel('N. Collision Checks')

    if scene == 'scene_1':
        plt.legend(loc='upper left')

    plt.savefig('../figs/' + scene + '_checks.png', bbox_inches='tight')


def plot_times(scene, checks):
    rrt_checks = checks['rrt']
    astar_checks = checks['astar']

    plt.figure()

    bpl = plt.boxplot(rrt_checks, positions=np.array(range(len(rrt_checks))) * 2.0 - 0.4, widths=0.5)
    bpr = plt.boxplot(astar_checks, positions=np.array(range(len(astar_checks))) * 2.0 + 0.4, widths=0.5)

    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    plt.plot([], c='#D7191C', label='RRT')
    plt.plot([], c='#2C7BB6', label='Hybrid A*')

    ticks = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '0.05', '0.1', '0.25', '0.5']

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)

    plt.xlabel(r'$p_{max}$')
    plt.ylabel('Planning Time (s)')

    if scene == 'scene_1':
        plt.legend(loc='upper left')

    plt.savefig('../figs/' + scene + '_times.png', bbox_inches='tight')


if __name__ == "__main__":
    argv = gflags.FLAGS(sys.argv)

    log_folder = gflags.FLAGS.log_folder
    scenes = ['scene_1', 'scene_2', 'scene_3']
    algs = ['rrt', 'astar']

    for scene in scenes:

        nodes = {}
        checks = {}
        times = {}

        for alg in algs:
            nodes[alg] = []
            checks[alg] = []
            times[alg] = []

            for prob in ['1e-05', '0.0001', '0.001', '0.01', '0.05', '0.1', '0.25', '0.5']:
                part_res_dir = "%s_%s_%s_%s" % (scene, alg, 'sprt', prob)

                for directory in os.listdir(log_folder):
                    if part_res_dir in directory:
                        res_dir = directory
                        break

                filepath = log_folder + "/" + res_dir + "/results.log"

                results = get_data_from_single_file(filepath)

                nodes[alg].append(results[3])
                checks[alg].append(results[5])
                times[alg].append(results[2])

        plot_nodes(scene, nodes)
        plot_checks(scene, checks)
        plot_times(scene, times)
