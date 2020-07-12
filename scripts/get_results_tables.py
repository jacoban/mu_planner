import gflags
import sys
import os
import numpy as np
from scipy import stats

from get_result_data import get_data_from_single_file

gflags.DEFINE_string('c_prob', '0.01', 'collision probability')
gflags.DEFINE_string('log_folder', '../log', 'log folder')

INDIFF_INTERVAL = dict([(0.01, 0.005), (0.1, 0.01), (0.25, 0.01)])


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h1 = se * stats.t.ppf((1 + confidence) / 2., n-1)

    return m, h1

def mean_std(data):
    return np.mean(data), np.std(data, ddof=1)

def write_latex_table_code(log_folder, c_prob, nodes, checks, solved, times, costs, p_coll):
    f = open(log_folder + "/table_prob_" + str(c_prob) + ".tex", "w")

    final_list = []
    for scene in ['scene_1', 'scene_2', 'scene_3']:
        for alg in ['rrt','astar']:
            for method in ['sprt','mc']:
                final_list += [solved[scene][alg][method], checks[scene][alg][method],
                times[scene][alg][method], p_coll[scene][alg][method]]

    final_tuple = tuple(final_list + [str(c_prob)])

    latex_string = """
    
\\begin{table}[t]
\\begin{scriptsize}
\\centering
\\begin{tabular}{|l|l|l|l|l|l|}
\hline
\multicolumn{2}{|l|}{MG-HU} & \\%% & Checks ($\\times 10^{6}$) & Time (s) & $p_{coll}(\pi)$ \\\ \hline
\multirow{2}{*}{RRT}  & SPRT  & %s & %s & %s & %s  \\\ \cline{2-6} 
                      &  MC & %s & %s & %s & %s  \\\ \hline
\multirow{2}{*}{HA*}  & SPRT  & %s & %s   & %s & %s \\\ \cline{2-6} 
                      &  MC & %s & %s & %s & %s   \\\ \hline
\multicolumn{2}{|l|}{MG-SU} &  &   & &  \\\ \hline
\multirow{2}{*}{RRT}  & SPRT  & %s &  %s  & %s & %s \\\ \cline{2-6} 
                      &  MC & %s & %s &  %s  & %s \\\ \hline
\multirow{2}{*}{HA*}  & SPRT &  %s  & %s & %s & %s \\\ \cline{2-6} 
                      &  MC & %s & %s &  %s  & %s \\\ \hline
\multicolumn{2}{|l|}{HG-HU}  &    &  &  & \\\ \hline
\multirow{2}{*}{RRT}  & SPRT &  %s  & %s & %s & %s \\\ \cline{2-6} 
                      &  MC & %s &  %s  & %s & %s \\\ \hline
\multirow{2}{*}{HA*}  & SPRT   & %s & %s & %s & %s \\\ \cline{2-6} 
                     & MC & %s &  %s  & %s & %s \\\ \hline
\end{tabular}
\caption{SPRT vs Naive MC, $p_{max}=%s$.}
\label{tab:my-table}
\\end{scriptsize}
\end{table}""" %final_tuple

    f.write(latex_string)
    f.close()


def write_latex_table_code_25(log_folder, c_prob, nodes, checks, solved, times, costs, p_coll):
    f = open(log_folder + "/table_prob_" + str(c_prob) + ".tex", "w")

    final_list = []
    for scene in ['scene_1', 'scene_2', 'scene_3', 'scene_4']:
        for alg in ['rrt', 'astar']:
            for method in ['sprt', 'mc']:
                final_list += [solved[scene][alg][method], checks[scene][alg][method],
                               times[scene][alg][method], p_coll[scene][alg][method]]

    final_tuple = tuple(final_list + [str(c_prob)])

    latex_string = """

\\begin{table}[t]
\\begin{scriptsize}
\\centering
\\begin{tabular}{|l|l|l|l|l|l|}
\hline
\multicolumn{2}{|l|}{MG-HU} & \\%% & Checks ($\\times 10^{6}$) & Time (s) & $p_{coll}(\pi)$ \\\ \hline
\multirow{2}{*}{RRT}  & SPRT  & %s & %s & %s & %s  \\\ \cline{2-6} 
                      &  MC & %s & %s & %s & %s  \\\ \hline
\multirow{2}{*}{HA*}  & SPRT  & %s & %s   & %s & %s \\\ \cline{2-6} 
                      &  MC & %s & %s & %s & %s   \\\ \hline
\multicolumn{2}{|l|}{MG-SU} &  &   & &  \\\ \hline
\multirow{2}{*}{RRT}  & SPRT  & %s &  %s  & %s & %s \\\ \cline{2-6} 
                      &  MC & %s & %s &  %s  & %s \\\ \hline
\multirow{2}{*}{HA*}  & SPRT &  %s  & %s & %s & %s \\\ \cline{2-6} 
                      &  MC & %s & %s &  %s  & %s \\\ \hline
\multicolumn{2}{|l|}{HG-HU}  &    &  &  & \\\ \hline
\multirow{2}{*}{RRT}  & SPRT &  %s  & %s & %s & %s \\\ \cline{2-6} 
                      &  MC & %s &  %s  & %s & %s \\\ \hline
\multirow{2}{*}{HA*}  & SPRT   & %s & %s & %s & %s \\\ \cline{2-6} 
                     & MC & %s &  %s  & %s & %s \\\ \hline

\multicolumn{2}{|l|}{BL}  &    &  &  & \\\ \hline
\multirow{2}{*}{RRT}  & SPRT &  %s  & %s & %s & %s \\\ \cline{2-6} 
                      &  MC & %s &  %s  & %s & %s \\\ \hline
\multirow{2}{*}{HA*}  & SPRT   & %s & %s & %s & %s \\\ \cline{2-6} 
                     & MC & %s &  %s  & %s & %s \\\ \hline

\end{tabular}
\caption{SPRT vs Naive MC, $p_{max}=%s$.}
\label{tab:my-table}
\\end{scriptsize}
\end{table}""" % final_tuple

    f.write(latex_string)
    f.close()


if __name__ == "__main__":
    argv = gflags.FLAGS(sys.argv)

    c_prob = gflags.FLAGS.c_prob

    scenes = ['scene_1', 'scene_2', 'scene_3']

    if c_prob == "0.25":
        scenes += ['scene_4']

    algs = ['rrt', 'astar']
    methods = ['sprt', 'mc']

    log_folder = gflags.FLAGS.log_folder

    nodes = {}
    checks = {}
    solved = {}
    times = {}
    costs = {}
    p_coll = {}

    for scene in scenes:
        nodes[scene] = {}
        checks[scene] = {}
        solved[scene] = {}
        times[scene] = {}
        costs[scene] = {}
        p_coll[scene] = {}

        for alg in algs:
            nodes[scene][alg] = {}
            checks[scene][alg] = {}
            solved[scene][alg] = {}
            times[scene][alg] = {}
            costs[scene][alg] = {}
            p_coll[scene][alg] = {}

            for method in methods:
                part_res_dir = "%s_%s_%s_%s" % (scene, alg, method, c_prob)

                res_dir = None
                for directory in os.listdir(log_folder):
                    if part_res_dir in directory:
                        res_dir = directory
                        break

                if res_dir is not None:

                    filepath = log_folder + "/" + res_dir + "/results.log"

                    ctot, cnot_solved, ctimes, cnodes, cchecks_total, cchecks_real, cpath_costs, cpath_pcoll = \
                        get_data_from_single_file(filepath)
                    perc_n_solved = (len(list(filter(lambda x: x == False, cnot_solved))) / float(ctot))

                    nodes[scene][alg][method] = "%d $\pm$ %d" %(mean_std(cnodes))
                    checks[scene][alg][method] = "%.2f $\pm$ %.2f" %(mean_std(cchecks_real)[0]/1e6,
                                                                   mean_std(cchecks_real)[1]/1e6)
                    solved[scene][alg][method] = "%d" %int(perc_n_solved*100)
                    times[scene][alg][method] = "%.1f $\pm$ %.1f" %mean_std(ctimes)
                    costs[scene][alg][method] = "%.1f $\pm$ %.1f" %mean_std(cpath_costs)

                    avg_p, std_dev_p = mean_std(cpath_pcoll)
                    p_coll[scene][alg][method] = "%.3f $\pm$ %.3f}" % (avg_p, std_dev_p)

                    if avg_p + std_dev_p <= float(c_prob):
                        p_coll[scene][alg][method] = "\\textcolor{green}{" + p_coll[scene][alg][method]

                    elif avg_p + std_dev_p <= float(c_prob) + INDIFF_INTERVAL[float(c_prob)]:
                        p_coll[scene][alg][method] = "\\textcolor{orange}{" + p_coll[scene][alg][method]

                    else:
                        p_coll[scene][alg][method] = "\\textcolor{red}{" + p_coll[scene][alg][method]

                else:
                    nodes[scene][alg][method] = "AAA"
                    checks[scene][alg][method] = "AAA"
                    solved[scene][alg][method] = "AAA"
                    times[scene][alg][method] = "AAA"
                    costs[scene][alg][method] = "AAA"
                    p_coll[scene][alg][method] = "AAA"

    if float(c_prob) < 0.2:
        write_latex_table_code(log_folder, c_prob, nodes, checks, solved, times, costs, p_coll)
    else:
        write_latex_table_code_25(log_folder, c_prob, nodes, checks, solved, times, costs, p_coll)
