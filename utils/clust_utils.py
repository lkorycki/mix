from pykeops.torch import LazyTensor
import torch


class ClustUtils:

    @staticmethod
    def k_means(x, k, n_iter=10):
        """Implements Lloyd's algorithm for the Euclidean metric.
        Stolen from: https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
        """

        n, d = x.shape

        c = x[:k, :].clone()
        cl = []

        x_i = LazyTensor(x.view(n, 1, d))  # (n, 1, d) samples
        c_j = LazyTensor(c.view(1, k, d))  # (1, k, d) centroids

        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(n_iter):

            # E step: assign points to the closest cluster -------------------------
            d_ij = ((x_i - c_j) ** 2).sum(-1)  # (n, k) symbolic squared distances
            cl = d_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, d), x)

            # Divide by the number of points per cluster:
            n_cl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
            c /= n_cl  # in-place division to compute the average

        return cl, c

    @staticmethod
    def nearest_cluster(x, c):
        d_ij = ((x - c) ** 2).sum(-1)
        return d_ij.argmin(dim=1).long().view(-1)
