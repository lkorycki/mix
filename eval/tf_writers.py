import tensorflow as tf
import numpy as np
import torch
import os
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

from core.clearn import ContinualExtractor
from learners.mix import MIX
from utils.plt_utils import PlotUtils as pu


class TBScalars:

    @staticmethod
    def write_epoch_result(tasks_acc, epoch, stream_label, i):
        tf.summary.scalar(f'{stream_label}#EPOCHS/C{i}', sum(tasks_acc) / len(tasks_acc), epoch,
                          description='x=epochs, y=overall accuracy')
        tf.summary.flush()

    @staticmethod
    def write_tasks_results(stream_label, tasks_acc, i):
        for j, acc in enumerate(tasks_acc):
            tf.summary.scalar(f'{stream_label}/C{j}', acc, i,
                              description='x=class ids, y=accuracy for a given Ci')  # todo: add configurable class-aggregation?

        tf.summary.scalar(f'ALL/{stream_label}', sum(tasks_acc) / len(tasks_acc), i,
                          description='x=class ids, y=overall accuracy')
        tf.summary.flush()

    @staticmethod
    def write_time_metrics(time_metrics, stream_label, i):
        tf.summary.scalar(f'TIME/{stream_label}/UPDATE', sum(time_metrics['update']) / len(time_metrics['update']), i,
                          description='x=class ids, y=average time')
        tf.summary.scalar(f'TIME/{stream_label}/PREDICTION', sum(time_metrics['prediction']) / len(time_metrics['prediction']), i,
                          description='x=class ids, y=average time')
        tf.summary.flush()


class TBImages:

    @staticmethod
    def write_test_data(data: Dataset, i: int, stream_label: str, cls_names: list):
        loader = DataLoader(data, batch_size=100, shuffle=True)

        images, labels = next(iter(loader))
        images = np.transpose(images.reshape(*images.shape), (0, 2, 3, 1))
        figure = pu.create_image_grid(images, labels, cls_names)

        tf.summary.image(f'{stream_label}#EXAMPLES', pu.fig_to_image(figure), step=i)
        tf.summary.flush()

    @staticmethod
    def write_confusion_matrices(labels, preds, i, stream_label):
        cm = sklearn.metrics.confusion_matrix(labels, preds)
        cm[np.isnan(cm)] = 0.0

        figure = pu.create_confusion_matrix(cm, class_names=[f'C{k}' for k in range(len(cm))])

        tf.summary.image(f'{stream_label}#CONF-MATS', pu.fig_to_image(figure), step=i)
        tf.summary.flush()

        return cm

    @staticmethod
    def write_embeddings_vis(data, model: ContinualExtractor, model_label, stream_label: str, i: int, n: int = 256,
                             epoch: bool = False, j: int = -1):
        x, y = [], []
        for loader in data.values():
            m = 0
            while m < n and loader:
                inputs, labels = next(iter(loader))
                k = min(n - m, len(inputs))
                features = model.extract(inputs).detach().cpu().numpy()
                x.append(features[:k])
                y.append(labels.numpy()[:k])
                m += len(inputs)

        centroids, covs = [], []
        if isinstance(model, IncrementalDeepGMM):
            mixtures = model.get_model().get_mixtures()
            for mixture in mixtures:
                cs, cv = [], []

                for component in mixture.get_components():
                    mean, cov = component.get_statistics()
                    cs.append(mean.clone().detach().cpu().numpy())
                    cv.append(torch.diag(cov).clone().detach().cpu().numpy())
                    #cv.append(cov.clone().detach().cpu().numpy())

                centroids.append(cs)
                covs.append(cv)

        centroids, covs = np.array(centroids), np.array(covs)
        cx = centroids.reshape(centroids.shape[0] * centroids.shape[1], centroids.shape[2])
        cy = np.array([[i] * len(cs) for i, cs in enumerate(centroids)]).flatten()
        x, y, cx, cy = np.concatenate(x), np.concatenate(y), cx, cy

        TBImages.write_raw(x, y, cx, cy, covs, model_label, stream_label, i, epoch, j)
        # TBImages.write_pca(x, y, cx, cy, stream_label, i, epoch, j)
        # TBImages.write_tsne(x, y, cx, cy, stream_label, i, epoch, j)

    @staticmethod
    def write_raw(x, y, cx, cy, covs, model_label, stream_label, i, epoch, j):
        figure = pu.create_scatter_plot(x, y, cx, cy, covs)
        # if j == -1:
        #     out_dir = f'runs/imgs_output/{model_label}-{stream_label}'
        #     os.makedirs(out_dir, exist_ok=True)
        #     figure.savefig(f'{out_dir}/vis_{i}.pdf', bbox_inches='tight')

        tf.summary.image(f'{stream_label}#EMBS-RAW{"-epochs/" + str(i) if epoch else ""}', pu.fig_to_image(figure),
                         step=j if epoch else i)
        tf.summary.flush()

    @staticmethod
    def write_pca(x, y, cx, cy, stream_label, i, epoch, j):
        pca = PCA(n_components=2)
        pca.fit(np.concatenate((x, cx)))
        pca.fit(x)
        x, cx = pca.transform(x), pca.transform(cx)

        figure = pu.create_scatter_plot(x, y, cx, cy, None)
        tf.summary.image(f'{stream_label}#EMBS-PCA{"-epochs/" + str(i) if epoch else ""}', pu.fig_to_image(figure),
                         step=j if epoch else i)
        tf.summary.flush()

    @staticmethod
    def write_tsne(x, y, cx, cy, stream_label, i, epoch, j):
        tsne = TSNE(n_components=2, random_state=0)
        f = tsne.fit_transform(np.concatenate((x, cx)))
        x, cx = f[:len(x)], f[len(x):]

        figure = pu.create_scatter_plot(x, y, cx, cy, None)
        tf.summary.image(f'{stream_label}#EMBS-TSNE{"-epochs/" + str(i) if epoch else ""}', pu.fig_to_image(figure),
                         step=j if epoch else i)
        tf.summary.flush()

    @staticmethod
    def write_epoch_loss(loss, epoch, stream_label, i):
        tf.summary.scalar(f'{stream_label}#LOSS-EPOCHS/C{i}', loss, epoch, description='x=epochs, y=loss')
        tf.summary.flush()

    @staticmethod
    def init():
        pu.register_new_cmaps()
        pu.register_alpha_cmaps()
