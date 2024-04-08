import collections
import csv
import shutil
import os
import fnmatch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib as mpl

mpl.use('pgf')
from utils.plt_utils import PlotUtils as pu

home = str(Path.home())
text_width = 433.62


def figsize(scale):
    fig_width_pt = 1.15 * text_width  # Get this from LaTeX using \the\textwidth (or column_width)
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width  # * golden_mean
    fig_size = [fig_width, fig_height]

    return fig_size


def init_pgf(output_scale):
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": ['Computer Modern'],
        "font.monospace": [],
        "axes.labelsize": int(16),
        "font.size": int(20),
        "legend.fontsize": int(22),
        "xtick.labelsize": int(20),
        "ytick.labelsize": int(20),
        "axes.titlesize": int(20),
        "figure.figsize": figsize(output_scale),
        "pgf.preamble": r"\usepackage{times}"
    }
    mpl.rcParams.update(pgf_with_latex)


def proc_params_results(param, dir_path, out_dir_path):
    data_dirs = next(os.walk(dir_path))[1]
    all_results = {'ALL': {'avg': defaultdict(list), 'series': defaultdict(list)}}

    for d in data_dirs:
        data_dir_path = f'{dir_path}/{d}'
        dirs = fnmatch.filter(os.listdir(data_dir_path), '*.csv')

        for f in dirs:
            _, params, data = parse_name(f)
            p = param_map(param)

            f = os.path.join(data_dir_path, f)
            if 'error' in f:
                avg = float('NaN')
                series = np.array([])
                data = data.replace('_error', '')
            else:
                results = collections.defaultdict(list)

                with open(f, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        results[row[0]] = [float(v) for v in row[1:]]

                all_row = results['0']
                avg = sum(all_row) / len(all_row)
                series = all_row

            pvs = '-'.join([params[x] for x in p])
            if data not in all_results:
                all_results[data] = {'avg': {}, 'series': {}}
            all_results[data]['avg'][pvs] = avg
            all_results[data]['series'][pvs] = series
            all_results['ALL']['avg'][pvs].append(avg)
            all_results['ALL']['series'][pvs].append(series)

    for k, v in all_results['ALL']['avg'].items():
        all_results['ALL']['avg'][k] = np.mean(np.array(v))
    for k, v in all_results['ALL']['series'].items():
        all_results['ALL']['series'][k] = np.mean(np.array(v), axis=0)

    for d, r in all_results.items():
        out_path = f'{out_dir_path}/{d}_{param}'
        os.makedirs(out_dir_path, exist_ok=True)

        print(f'Writting to: {out_path}')
        if param == 'tightness':
            proc_tightness(r, d, out_path)
        if param == 'learning_rates':
            proc_learning_rates(r, d, out_path)
        elif param == 'loss_updates':
            proc_loss_updates(r, d, out_path)
        elif param == 'replay_buffers':
            proc_replay_buffers(r, d, out_path)
        elif param == 'ks':
            proc_ks(r, d, out_path)
        elif param == 'covs':
            proc_covs(r, d, out_path)


def param_map(param_name):
    m = {
        'learning_rates': ['lr'],
        'tightness': ['t'],
        'loss_updates': ['lt', 'clasm'],
        'covs': ['fc'],
        'replay_buffers': ['rbs'],
        'ks': ['k']
    }
    return m[param_name]


def parse_name(f_name):
    f_name = f_name.replace('e-0', 'e0')
    param, label, data = f_name.split('#')
    return param, {k: v for k, v in [p.split('=') for p in label.split('-')[1:]]}, data.replace('.csv', '')


def proc_tightness(results, d, out_path):
    inter = ['1e06', '1e05', '0.0001', '0.001', '0.01']
    intra = ['1e06', '1e05', '0.0001', '0.001', '0.01']
    lia, lie = len(intra), len(inter)

    grid = np.zeros((lia, lie))
    for i, a in enumerate(intra):
        for j, e in enumerate(inter):
            grid[lie - 1 - i][j] = results['avg'][f'{e}x{a}']

    init_pgf(0.32)
    plot_fig(pu.create_grid(grid,
                            ['1e-06', '1e-05', '1e-04', '1e-03', '1e-02'],
                            ['1e-06', '1e-05', '1e-04', '1e-03', '1e-02'][::-1],
                            'inter', 'intra',
                            title=d.split('-')[0],
                            mi=np.min(grid),
                            mx=np.max(grid)),
             out_path)


def proc_learning_rates(results, d, out_path):
    ex_lrs = ['1e07', '1e06', '1e05', '0.0001', '0.001']
    gmm_lrs = ['1e05', '0.0001', '0.001', '0.01', '0.1']
    lg, le = len(gmm_lrs), len(ex_lrs)

    grid = np.zeros((lg, le))
    for i, g in enumerate(gmm_lrs):
        for j, e in enumerate(ex_lrs):
            grid[lg - 1 - i][j] = results['avg'][f'{e}x{g}']

    init_pgf(0.32)
    plot_fig(pu.create_grid(grid,
                            ['1e-07', '1e-06', '1e-05', '1e-04', '1e-03'],
                            ['1e-05', '1e-04', '1e-03', '1e-02', '1e-01'][::-1],
                            r'$\alpha_{F}$', r'$\alpha_{G}$',
                            title=d.split('-')[0],
                            mi=np.min(grid),
                            mx=np.max(grid)),
             out_path)


def plot_fig(figure, out_dir_path):
    print('Writing to: ', f'{out_dir_path}.pdf')
    figure.savefig(f'{out_dir_path}.pdf', bbox_inches='tight')
    plt.close(figure)


def proc_loss_updates(results, d, out_path):
    order_cls = ['softmax', 'max_component']
    order_update = ['ce', 'mp', 'mpr']
    m = {'ce': 'CE', 'mp': 'MC', 'mpr': 'MCR', 'max_component': 'max-component', 'softmax': 'softmax'}

    bars = []
    for c in order_cls:
        c_bars = []
        for u in order_update:
            avg = results['avg'][f'{u}-{c}']
            c_bars.append(avg)

        bars.append(c_bars)

    init_pgf(0.3)
    print(d)
    plot_fig(
        pu.create_bar_plot(bars, ['softmax', 'max-component'], [m[u] for u in order_update], '', '', d.split('-')[0],
                           d == 'IMAGENET10-CI-1.0', (8, 6)),
        out_path)


def proc_replay_buffers(results, d, out_path):
    order = [8, 64, 128, 256, 512]

    all_series = []
    for v in order:
        series = results['series'][str(v)]
        all_series.append(series)

    init_pgf(0.3)
    plot_fig(
        pu.create_line_plot(all_series, order, '', '', d.split('-')[0], d == 'ALL', (8, 6), False),
        out_path)


def proc_ks(results, d, out_path):
    order = [1, 3, 5, 10, 20]
    out = open(f'{out_path}-{d}', 'w+')
    bars = ''

    for k in order:
        avg = results['avg'][f'{k}']
        bars += '{:.4f}\n'.format(avg)

    out.write(f'{d}-ks\n')
    out.write(f'{bars}\n')

    out.close()


def proc_covs(results, d, out_path):
    order = ['False', 'True']
    out = open(f'{out_path}-{d}', 'w+')

    for v in order:
        avg = results['avg'][v]
        out.write(f'{d}-rep-{v}\n')
        out.write('{:.4f}\n'.format(avg))

    out.close()


def proc_final_results(dir_path, out_dir_path, algs, offline=True):
    data_dirs = next(os.walk(dir_path))[1]
    all_results = {'ALL': {'avg': defaultdict(list), 'series': defaultdict(list)}}

    for alg in data_dirs:
        if alg not in algs:
            continue

        data_dir_path = f'{dir_path}/{alg}'
        dirs = fnmatch.filter(os.listdir(data_dir_path), '*.csv')

        for f in dirs:
            data = parse_final_name(f)
            if 'IMAGENET200-PRE1000-CI-1.0' in data:
                continue

            f = os.path.join(data_dir_path, f)
            if 'error' in f:
                avg = float('NaN')
                series = []
                data = data.replace('_error', '')
            else:
                results = collections.defaultdict(list)

                with open(f, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        results[row[0]] = [float(v) for v in row[1:]]

                all_row = results['0']
                avg = sum(all_row) / len(all_row)
                series = all_row

            if data not in all_results:
                all_results[data] = {'avg': defaultdict(list), 'series': defaultdict(list)}
            all_results[data]['avg'][alg] = avg
            all_results[data]['series'][alg] = series
            all_results['ALL']['avg'][alg].append(avg)

    for k, v in all_results['ALL']['avg'].items():
        all_results['ALL']['avg'][k] = np.mean(np.array(v))

    for d, r in all_results.items():
        os.makedirs(out_dir_path + '/tables', exist_ok=True)

        print(f'Writting to: {out_dir_path}')
        proc_final(r, d, algs, out_dir_path, offline)


def parse_final_name(f_name):
    data = f_name.split('#')[-1]
    return data.replace('.csv', '')


offline_map = {
    'MNIST-CI-1.0': 1.0,
    'FASHION-CI-1.0': 0.9865,
    'SVHN-CI-1.0': 1.0,
    'CIFAR10-CI-1.0': 1.0,
    'IMAGENET10-CI-1.0': 1.0,
    'CIFAR20-CI-1.0': 0.7635,
    'IMAGENET20A-CI-1.0': 0.7180,
    'IMAGENET20B-CI-1.0': 0.7929,
    'CIFAR100-PRE10-CI-1.0': 0.1572,
    'CIFAR100-PRE100-CI-1.0': 0.5955,
    'IMAGENET200-PRE20B-CI-1.0': 0.1838,
    'IMAGENET200-PRE200-CI-1.0': 0.6702

}


def proc_final(results, d, order, out_path, offline=True):
    out = open(f'{out_path}/tables/{d}', 'w+')
    all_series = []
    m = {'ig-ce': 'MIX-CE', 'ig-mpr': 'MIX-MCR', 'agem': 'A-GEM', 'icarl': 'iCaRL'}

    for alg in order:
        avg = results['avg'][alg]
        out.write(f'{d}-{alg}: ' + '{:.4f}\n'.format(avg))
        series = results['series'][alg]
        ln = len(series)

        if (ln == 20) or (ln == 10):
            series = [1.0] + series
        all_series.append(series[1:])

    out.close()

    if d != 'ALL':
        plot_legend = (d in ['MNIST-CI-1.0'])
        ds = d.split('-')
        t = ds[0] if 'PRE' not in ds[1] else f'{ds[0]}-{ds[1]}'
        init_pgf(0.49)
        plot_fig(
            pu.create_line_plot(all_series, [m[alg] if (alg in m) else alg.upper() for alg in order], '', '', t,
                                plot_legend, (10, 8), True, offline_map[d] if offline else None),
            f'{out_path}/{d}-series')


def move_mix_final_results_files(results_dir, out_dir):
    data_dirs = next(os.walk(results_dir))[1]

    for data in data_dirs:
        data_dir_path = f'{results_dir}/{data}'

        results = fnmatch.filter(os.listdir(data_dir_path), '*.csv')
        for r in results:
            v = r.split('-')[0]
            os.makedirs(f'{out_dir}/{v}/', exist_ok=True)
            shutil.copy(f'{data_dir_path}/{r}', f'{out_dir}/{v}/')

        cms = fnmatch.filter(os.listdir(data_dir_path + '/cms'), '*.npy')
        for cm in cms:
            v = cm.split('-')[0]
            os.makedirs(f'{out_dir}/{v}/cms/', exist_ok=True)
            shutil.copy(f'{data_dir_path}/cms/{cm}', f'{out_dir}/{v}/cms/')


def move_mix_final_runs_files(runs_dir, out_dir):
    data_dirs = next(os.walk(runs_dir))[1]

    for data in data_dirs:
        data_dir_path = f'{runs_dir}/{data}'

        runs = next(os.walk(data_dir_path))[1]
        for r in runs:
            v = r.split('-')[0]
            os.makedirs(f'{out_dir}/{v}/', exist_ok=True)
            shutil.copytree(f'{data_dir_path}/{r}', f'{out_dir}/{v}/{data}-{v}')


def run():
    print('Processing results')
    prefix = 'mix'

    params = ['learning_rates', 'tightness', 'loss_updates', 'replay_buffers', 'covs', 'ks']
    for param in params:
        proc_params_results(
            param,
            f'{home}/Results/{prefix}/{prefix}-params/results/{param}',
            f'{home}/Results/{prefix}/{prefix}-params/proc-results/{param}')

    move_mix_final_results_files(f'{home}/Downloads/{prefix}-final-results', f'{home}/Downloads/{prefix}-final-results-proc')
    move_mix_final_runs_files(f'{home}/Downloads/{prefix}-final-runs', f'{home}/Downloads/{prefix}-final-runs-proc')

    proc_final_results(
        f'{home}/Results/{prefix}/{prefix}-baselines-final/results',
        f'{home}/Results/{prefix}/{prefix}-baselines-final/proc-results',
        ['ig-mpr', 'ig-ce', 'er', 'ersb', 'icarl', 'gss', 'der', 'agem', 'lwf', 'si', 'naive'])

    proc_final_results(
        f'{home}/Results/{prefix}/{prefix}-final-fixed/results',
        f'{home}/Results/{prefix}/{prefix}-final-fixed/proc-results',
        ['ig-mpr', 'ig-ce', 'er', 'ersb', 'icarl', 'gss', 'der', 'lwf', 'si', 'naive'],
        offline=True)


if __name__ == '__main__':
    run()
