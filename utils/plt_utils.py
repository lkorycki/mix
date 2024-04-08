import torch

import matplotlib.pyplot as plt
import io
import tensorflow as tf
import itertools
import numpy as np
import matplotlib.colors as mcolors
from scipy.stats import multivariate_normal
import matplotlib
from scipy.ndimage.filters import gaussian_filter1d


class PlotUtils:

    ocean = (0.0, 2*61/256, 2*78/256)
    colors = np.array(['red', 'blue', 'green', 'orange', 'grey', 'brown', ocean, 'purple', 'cyan', 'magenta', 'black',
                       'yellow', 'tan'],
                      dtype=object)
    cmps = np.array([c + "_alpha" for c in
                     ['Reds', 'Blues', 'Greens', 'Oranges', 'Greys', 'Browns', 'Oceans', 'Purples', 'Cyans', 'Magentas']])

    latex_colors = {
        'green1': '#004c00',
        'green2': '#009900',
        'green3': '#47da47',
        'green4': '#99ea99',
        'blue1': '#003566',
        'blue2': '#186fc0',
        'blue3': '#68aae7',
        'blue4': '#a1caf0',
        'red0': '#990000',
        'red1': '#DD0000',
        'red2': '#ea6262',
        'red3': '#F19999',
        'red4': '#f9d6d6',
        'yellow1': '#f1b139',
        'gray1': '#C8C8C8',
        'gray2': '#707070',
        'gray3': '#505050',
        'brown1': '#654321'
    }


    @staticmethod
    def create_image_grid(images, labels, cls_names):
        figure = plt.figure(figsize=(20, 20))
        rows, cols = (10, 10) if len(images) == 100 else (5, 5)

        for i in range(min(rows * cols, len(images))):
            cls_idx = labels[i].item()
            plt.subplot(rows, cols, i + 1, title=cls_names[cls_idx] if len(cls_names) > cls_idx else cls_idx)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])

        return figure

    @staticmethod
    def fig_to_image(figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)

        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image

    @staticmethod
    def create_confusion_matrix(cm, class_names, title=None):
        figure = plt.figure(figsize=(8, 8))

        colors = plt.cm.Blues(np.linspace(0, 1, 128))
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

        plt.imshow(cm, interpolation='nearest', cmap=cmap if len(cm) > 1 else plt.cm.Blues_r)
        if title:
            plt.title(title, fontsize=24, pad=16)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)

        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, labels[i, j], horizontalalignment='center', color=color)

        plt.ylabel('True label', fontsize=18, labelpad=20)
        plt.xlabel('Predicted label', fontsize=18, labelpad=20)

        return figure

    @staticmethod
    def create_grid(grid, x_ticks, y_ticks, x_label, y_label, title=None, mi=None, mx=None):
        figure = plt.figure(figsize=(8, 8))

        # colors = plt.cm.RdYlGn(np.linspace(0, 1, 128))
        cmap = 'RdYlGn' # mcolors.LinearSegmentedColormap.from_list('colormap', colors)

        if mi is None or mx is None:
            plt.imshow(grid, interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(grid, interpolation='nearest', cmap=cmap, vmin=mi, vmax=mx)

        if title:
            plt.title(title, fontsize=42, pad=16)

        tick_marks = np.arange(len(x_ticks))
        plt.xticks(tick_marks, x_ticks)
        plt.yticks(tick_marks, y_ticks)
        ax = plt.gca()
        ax.tick_params(axis='y', which='major', pad=15, labelsize=30)
        ax.tick_params(axis='x', which='major', pad=15, labelsize=26)

        labels = np.around(grid.astype('float'), decimals=2)

        for i, j in itertools.product(range(grid.shape[0]), range(grid.shape[1])):
            color = 'black'
            plt.text(j, i, labels[i, j], horizontalalignment='center', color=color, fontdict={'size': 28})

        plt.ylabel(y_label, fontsize=40, labelpad=20)
        plt.xlabel(x_label, fontsize=40, labelpad=20)

        return figure

    @staticmethod
    def create_scatter_plot(x, y, cx, cy, covs):
        figure = plt.figure(figsize=(8, 8))

        if covs is not None:
            num_mixtures, num_components = len(covs), len(covs[0])
            cxr = cx.reshape(num_mixtures, num_components, cx.shape[-1])

            x1_mi, x1_mx = min(-0.35, min(x[:, 0])), max(0.35, max(x[:, 0]))
            x2_mi, x2_mx = min(-0.35, min(x[:, 1])), max(0.35, max(x[:, 1]))

            for i in range(num_mixtures):
                for j in range(num_components):
                    mean, cov = cxr[i][j], covs[i][j]
                    cov1_max = max(cov[:, 0])
                    cov2_max = max(cov[:, 1])
                    x1_left, x1_right = mean[0] - 5 * cov1_max, mean[0] + 5 * cov1_max
                    x2_left, x2_right = mean[1] - 5 * cov2_max, mean[1] + 5 * cov2_max

                    x1_mi, x1_mx = min(x1_mi, x1_left), max(x1_mx, x1_right)
                    x2_mi, x2_mx = min(x2_mi, x2_left), max(x2_mx, x2_right)

            for i in range(num_mixtures):
                for j in range(num_components):
                    mean, cov = cxr[i][j], covs[i][j]
                    PlotUtils.draw_contour(mean, cov, x1_mi, x1_mx, x2_mi, x2_mx, PlotUtils.cmps[i])

        plt.scatter(x[:, 0], x[:, 1], c=PlotUtils.colors[y.astype(np.int)], s=5, alpha=0.25)
        plt.scatter(cx[:, 0], cx[:, 1], c=PlotUtils.colors[cy.astype(np.int)], marker='x', s=50, alpha=1.0)

        plt.ylabel('x1', fontsize=18, labelpad=20)
        plt.xlabel('x2', fontsize=18, labelpad=20)

        return figure

    @staticmethod
    def create_bar_plot(data, bar_names, x_ticks, x_label, y_label, title=None, legend=False, figsize=(8, 8)):
        figure = plt.figure(figsize=figsize)
        x, s = np.arange(len(x_ticks)), 0.3

        ax = plt.gca()
        ax.grid(color='gray', linestyle='dashed', linewidth=2.0, alpha=0.2, axis='y', zorder=0)

        lc = PlotUtils.latex_colors
        colors = [lc['red1'], lc['green2'], lc['blue2'], lc['yellow1'], lc['gray1'], lc['brown1']]

        for i, d in enumerate(data):
            plt.bar(x + i * s, data[i], color=colors[i], width=0.25, zorder=3, alpha=1.0)

        if title:
            plt.title(title, fontsize=42, pad=16)

        tick_marks = np.arange(len(x_ticks)) + 0.5 * s
        plt.xticks(tick_marks, x_ticks)
        ax.tick_params(axis='y', which='major', pad=15, labelsize=30)
        ax.tick_params(axis='x', which='major', pad=15, labelsize=26)

        plt.ylim(0.0, 1.02)
        plt.ylabel(y_label, fontsize=40, labelpad=20)
        plt.xlabel(x_label, fontsize=40, labelpad=20)

        if legend:
            plt.legend(labels=bar_names, loc='upper left', bbox_to_anchor=(0.23, 1.0))

        return figure

    @staticmethod
    def create_line_plot(data, series_names, x_label, y_label, title=None, legend=False, figsize=(8, 8), final=True,
                         max_point=None):
        figure = plt.figure(figsize=figsize)

        ax = plt.gca()
        ax.grid(color='gray', linestyle='dashed', linewidth=2.0, alpha=0.2, axis='y', zorder=0)

        lc = PlotUtils.latex_colors
        colors = PlotUtils.colors[:len(series_names)][::-1] if final else [lc['red4'], lc['red3'], lc['red2'], lc['red1'], lc['red0']]
        if final:
            data = data[::-1]

        x_len = len(data[0])
        sl = len(series_names)

        for i, d in enumerate(data):
            x = np.arange(x_len)
            d_smooth = gaussian_filter1d(d, sigma=0.5)
            lw = 5 if series_names[sl - i - 1] in ['MIX-CE', 'MIX-MCR'] else 4
            mk = 's' if series_names[sl - i - 1] == 'NAIVE' else 'o'
            plt.plot(x, d_smooth, marker=mk, markersize=7, color=colors[i], linewidth=lw)

        if max_point is not None:
            plt.plot(x_len - 1, max_point, marker='*', markersize=10, color='black', linewidth=0)

        if title:
            plt.title(title, fontsize=42, pad=16)

        x_ticks = np.arange(x_len, step=1 if x_len == 10 else 2)
        x_ticks_labels = x_ticks + 1
        if 'CIFAR100' in title or 'IMAGENET200' in title:
            x_ticks_labels = [int(x * 10) for x in x_ticks_labels]

        plt.xticks(x_ticks, x_ticks_labels)
        ax.tick_params(axis='y', which='major', pad=15, labelsize=30)
        ax.tick_params(axis='x', which='major', pad=15, labelsize=26)

        plt.ylabel(y_label, fontsize=40, labelpad=20)
        plt.xlabel(x_label, fontsize=40, labelpad=20)

        series_names = series_names[::-1]
        if legend:
            if max_point is not None:
                series_names.append('OFFLINE')

            plt.legend(labels=series_names, loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=2 if final else 1)

        return figure

    @staticmethod
    def draw_contour(mean, cov, x1_mi, x1_mx, x2_mi, x2_mx, color_map):
        x = np.linspace(x1_mi, x1_mx, 100)
        y = np.linspace(x2_mi, x2_mx, 100)

        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        mvn = torch.distributions.MultivariateNormal(torch.tensor(mean).float(), torch.tensor(cov).float())
        pp = torch.exp(mvn.log_prob(torch.tensor(pos).float())).numpy()
        plt.contour(x, y, pp, cmap=color_map)

        # mvn = torch.distributions.MultivariateNormal(torch.tensor(mean).float(), scale_tril=torch.tril(torch.tensor(cov).float()))
        # plt.contour(x, y, torch.exp(mvn.log_prob(torch.tensor(pos).float())).numpy(), cmap=color_map)

    @staticmethod
    # stolen from: https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
    def cmap_white2alpha(name, ensure_increasing=False):
        cmap = plt.get_cmap(name)

        rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
        rgba = PlotUtils.rgb_white2alpha(rgb, ensure_increasing=ensure_increasing)
        cmap_alpha = matplotlib.colors.ListedColormap(rgba, name=name + "_alpha")

        return cmap_alpha

    @staticmethod
    def rgb_white2alpha(rgb, ensure_increasing=False):
        alpha = 1. - np.min(rgb, axis=1)

        if ensure_increasing:
            a_max = alpha[0]
            for i, a in enumerate(alpha):
                alpha[i] = a_max = np.maximum(a, a_max)

        alpha = np.expand_dims(alpha, -1)
        # rgb = (rgb + alpha - 1) / alpha

        return np.concatenate((rgb, alpha), axis=1)

    @staticmethod
    def register_new_cmaps():
        def browns():
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(1, 90/256, 256)
            vals[:, 1] = np.linspace(1, 39/256, 256)
            vals[:, 2] = np.linspace(1, 41/256, 256)
            return mcolors.ListedColormap(vals, name='Browns')

        def oceans():
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(1, 0/256, 256)
            vals[:, 1] = np.linspace(1, 61/256, 256)
            vals[:, 2] = np.linspace(1, 78/256, 256)
            return mcolors.ListedColormap(vals, name='Oceans')

        def cyans():
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(1, 0/256, 256)
            vals[:, 1] = np.linspace(1, 196/256, 256)
            vals[:, 2] = np.linspace(1, 196/256, 256)
            return mcolors.ListedColormap(vals, name='Cyans')

        def magentas():
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(1, 196/256, 256)
            vals[:, 1] = np.linspace(1, 0/256, 256)
            vals[:, 2] = np.linspace(1, 196/256, 256)
            return mcolors.ListedColormap(vals, name='Magentas')

        plt.register_cmap(cmap=browns())
        plt.register_cmap(cmap=oceans())
        plt.register_cmap(cmap=cyans())
        plt.register_cmap(cmap=magentas())

    @staticmethod
    def register_alpha_cmaps():
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Reds'), name='Reds_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Greens'), name='Greens_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Blues'), name='Blues_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Oranges'), name='Oranges_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Greys'), name='Greys_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Browns'), name='Browns_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Oceans'), name='Oceans_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Purples'), name='Purples_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Cyans'), name='Cyans_alpha')
        plt.register_cmap(cmap=PlotUtils.cmap_white2alpha('Magentas'), name='Magentas_alpha')

