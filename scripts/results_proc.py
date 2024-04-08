import collections
import csv
import os
import fnmatch
import numpy as np
from matplotlib import pyplot as plt

from utils.plt_utils import PlotUtils as pu


def proc_results(dir_path: str):
    dirs = fnmatch.filter(os.listdir(dir_path), '*.csv')
    out = open(f'{dir_path}/output_avg', 'w+')

    for f in dirs:
        f = os.path.join(dir_path, f)
        print(f)

        results = collections.defaultdict(list)

        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                results[row[0]] = [float(v) for v in row[1:]]

        all_row = results['0']
        avg = sum(all_row) / len(all_row)
        series_line = ' '.join([f'{(i, r)}' for i, r in enumerate(all_row)])

        print(all_row, avg, series_line)

        out.write(f'{f}\n')
        out.write('{:.4f}\n'.format(avg))
        out.write(f'{series_line}\n\n')

    out.close()


def proc_retention(dir_path: str):
    dirs = fnmatch.filter(os.listdir(dir_path), '*.csv')
    out = open(f'{dir_path}/output_ret', 'w+')

    for f in dirs:
        f = os.path.join(dir_path, f)
        print(f)

        results = collections.defaultdict(list)

        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                results[row[0]] = [float(v) for v in row[1:]]

        ret_vals = [0.0, 0.0, 0.0]
        ret_cnts = [0, 0, 0]
        ks = [0, 5, 10] if len(results['0']) == 20 else [0, 2, 5]

        for k, vals in results.items():
            k = int(k) - 1
            if k == 0: continue

            print(k, vals)
            ret_vals[ks[0]] += vals[k]
            ret_cnts[ks[0]] += 1

            if k + ks[1] < len(vals):
                ret_vals[1] += vals[k + ks[1]]
                ret_cnts[1] += 1

            if k + ks[2] < len(vals):
                ret_vals[2] += vals[k + ks[2]]
                ret_cnts[2] += 1

        for i in range(len(ret_vals)):
            ret_vals[i] /= ret_cnts[i]

        bar_line = ' '.join([f'(+{ks[i]},{v})' for i, v in enumerate(ret_vals)])
        print(bar_line)

        out.write(f'{f}\n')
        out.write(f'{bar_line}\n\n')


import matplotlib as mpl
mpl.use('pgf')

output_scale = 0.33
column_width = 243.911
text_width = 347.123


def figsize(scale):
    fig_width_pt = 1.15*text_width  # Get this from LaTeX using \the\textwidth (or column_width)
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width  # * golden_mean
    fig_size = [fig_width, fig_height]

    return fig_size


def init_pgf():
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": ['Computer Modern'],
        "font.monospace": [],
        "axes.labelsize": int(16),
        "font.size": int(20),
        "legend.fontsize": int(6),
        "xtick.labelsize": int(20),
        "ytick.labelsize": int(20),
        "axes.titlesize": int(20),
        "figure.figsize": figsize(output_scale),
        "pgf.preamble": [
            r"\usepackage{times}"
        ]
    }
    mpl.rcParams.update(pgf_with_latex)


def create_new_fig():
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    return fig, ax


def proc_cms(dir_path: str, title: str):
    dirs = fnmatch.filter(os.listdir(dir_path), '*.npy')
    cm_agg = np.zeros((10, 10))

    for f in dirs:
        f = os.path.join(dir_path, f)
        print(f)

        cms = np.load(f, allow_pickle=True)
        cm10 = np.array(cms[0], dtype=np.float32)
        cm10 = cm10 / np.sum(cm10)
        cm_agg = np.add(cm_agg, cm10)

    print('Writing to: ', f'{dir_path}/{dir_path}.pdf')

    init_pgf()
    figure = pu.create_confusion_matrix(cm_agg, class_names=[f'C{k}' for k in range(len(cm_agg))], title=title)
    figure.savefig(f'{dir_path}.pdf', bbox_inches='tight')
    plt.close(figure)

