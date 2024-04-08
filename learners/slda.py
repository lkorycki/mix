from torch import Tensor

from core.clearn import ContinualLearner

import torch
import torch.nn as nn


class StreamingLDA(ContinualLearner):

    def __init__(self, input_shape, num_classes, test_batch_size=1024, shrinkage_param=1e-4,
                 streaming_update_sigma=True, device='cpu'):
        super().__init__()
        self.slda = _StreamingLDA(input_shape, num_classes, test_batch_size, shrinkage_param, streaming_update_sigma, device)

    def initialize(self, x_batch, y_batch, **kwargs):
        self.slda.fit_base(x_batch, y_batch)

    def predict(self, x_batch):
        outputs = self.slda.predict(x_batch)
        return torch.max(outputs, 1)[1]

    def predict_prob(self, x_batch):
        return self.slda.predict(x_batch, True)

    def update(self, x_batch: Tensor, y_batch: Tensor, **kwargs):
        for x, y in zip(*(x_batch, y_batch)):
            self.slda.update(x, y)


class _StreamingLDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant Analysis algorithm for streaming learning.
    Source: https://github.com/tyler-hayes/Deep_SLDA
    Reference:
        @InProceedings{Hayes_2020_CVPR_Workshops,
            author = {Hayes, Tyler L. and Kanan, Christopher},
            title = {Lifelong Machine Learning With Deep Streaming Linear Discriminant Analysis},
            booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
            month = {June},
            year = {2020}
        }
    """

    def __init__(self, input_shape, num_classes, test_batch_size=1024, shrinkage_param=1e-4, streaming_update_sigma=True,
                 device='cuda'):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param test_batch_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        """

        super(_StreamingLDA, self).__init__()

        # SLDA parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    def update(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.float().to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():

            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

            # update class means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1

    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.float().to(self.device)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                #print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                        self.device))
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        X = X.float().to(self.device)
        y = y.squeeze().long()

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)
