import random
from operator import itemgetter

from torch import nn
import torch
from torch.nn.functional import one_hot
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from core.clearn import ContinualLearner, ContinualExtractor
from torch.optim import Adam, SGD

from data.data_utils import IndexDataset


class IncrementalNearestPrototypes(ContinualLearner, ContinualExtractor):

    def __init__(self, extractor: nn.Module, k: int, extractor_lr: float, epochs: int, device='cpu'):
        super().__init__()
        self.extractor = extractor.to(device)
        self.raw_prototypes = Tensor()
        self.extractor_optimizer = SGD(self.extractor.parameters(), lr=extractor_lr) \
            if len(list(self.extractor.parameters())) else None
        self.extractor_lr = extractor_lr
        self.k = k
        self.alpha = 1.0
        self.epochs = epochs
        self.device = device

    def predict(self, x_batch):
        x_batch = x_batch.to(self.device)
        preds = torch.max(self.predict_prob(x_batch), 1)[1]

        return preds

    def predict_prob(self, x_batch):
        self.extractor.eval()
        with torch.no_grad():
            tsd_dists = self.prototypes_dist(
                self.extractor(x_batch.to(self.device)),
                self.extract_prototypes(self.raw_prototypes)
            )
        min_dists = torch.min(-tsd_dists, -1).values

        return -min_dists

    def extract_prototypes(self, prototypes):
        return self.extractor(torch.flatten(prototypes, start_dim=0, end_dim=1))

    def prototypes_dist(self, x_batch, prototypes):
        n, c, k, d = len(x_batch), len(self.raw_prototypes), self.k, len(x_batch[0])

        x_batch = x_batch.unsqueeze(1).unsqueeze(1).expand(n, c, k, d)
        prototypes = prototypes.reshape(c, k, d).unsqueeze(0).expand(n, c, k, d)  # todo: broadcast should work here auto?

        return self.tsd_dist(self.l2_dist(x_batch, prototypes), self.alpha)  # [bs, c, k]

    @staticmethod
    def l2_dist(a, b):
        l2 = torch.pow(a - b, 2).sum(-1)
        return l2

    @staticmethod
    def tsd_dist(l2_dists, alpha):
        # tsd = torch.pow(1.0 + l2_dists / alpha, -(alpha + 1.0) / 2.0)
        tsd = torch.exp(-l2_dists)
        # tsd = torch.tanh(l2_dists)

        # print(l2_dists, tsd)
        return tsd

    def update(self, x_batch, y_batch, **kwargs):
        class_batch = IndexDataset(TensorDataset(x_batch, y_batch))

        if len(self.raw_prototypes) > 0:
            # self.extractor.eval()
            # with torch.no_grad():
            #     prev_prototypes = self.extract_prototypes(self.raw_prototypes).detach()  # todo: no grad? eval?

            self.extractor.train()
            for _ in range(self.epochs):
                for xb, _, ib in DataLoader(class_batch, batch_size=128, shuffle=True):
                    if len(xb) <= 1:
                        continue
                    xb = self.extractor(xb.to(self.device))
                    prototypes = self.extract_prototypes(self.raw_prototypes)

                    # todo: add (proactive) compactness_loss?
                    ctr_loss = self.contrastive_loss(xb, prototypes)
                    # stb_loss = self.stability_loss(prototypes, prev_prototypes)
                    loss = ctr_loss # + stb_loss
                    # print(loss)
                    loss.backward()

                    with torch.no_grad():
                        self.extractor_optimizer.step()
                        self.extractor_optimizer.zero_grad()

        self.extractor.eval()
        with torch.no_grad():
            class_prototypes = self.generate_prototypes(class_batch)
            self.add_prototypes(class_prototypes)
            print(self.raw_prototypes.shape)

    def contrastive_loss(self, x_batch, prototypes):
        tsd_dists = self.prototypes_dist(x_batch, prototypes)
        # tsd_dists = torch.flatten(tsd_dists, start_dim=1)
        # mins = torch.max(tsd_dists, dim=0).values.detach()
        # tsd_dists = tsd_dists / mins
        # print(tsd_dists.shape, tsd_dists[:5])
        return torch.mean(tsd_dists)

        # return -torch.mean(self.prototypes_dist(x_batch, prototypes))

    def stability_loss(self, prototypes, prev_prototypes):
        return torch.mean(self.tsd_dist(self.l2_dist(prototypes, prev_prototypes), self.alpha))

    def generate_prototypes(self, class_data):
        x_batch, x_batch_features = [], []
        for xb, _, _ in DataLoader(class_data, batch_size=128, shuffle=False):
            x_batch.append(xb)
            x_batch_features.append(self.extract(xb.to(self.device)).cpu())

        x_batch, x_batch_features = torch.cat(x_batch), torch.cat(x_batch_features)
        # return self.herding_selection(x_batch, x_batch_features)
        return self.rand_selection(x_batch)

    def herding_selection(self, x_batch, x_batch_features):
        x_batch, x_batch_features = x_batch.numpy(), x_batch_features.numpy()
        x_batch_all = list(zip(x_batch, x_batch_features))

        batch_mean = np.mean(x_batch_features, axis=0)
        exemplars_sum = np.zeros(batch_mean.shape[-1])

        k = min(self.k, len(x_batch_all))
        class_prototypes = np.empty((k, *x_batch.shape[1:]))

        for k in range(k):
            min_idx, _ = min([(i, np.linalg.norm((x_feats + exemplars_sum) / (k + 1) - batch_mean))
                              for i, (x, x_feats) in enumerate(x_batch_all)], key=itemgetter(1))
            class_prototypes[k] = x_batch_all[min_idx][0]  # raw prototype
            exemplars_sum += x_batch_all[min_idx][1]
            del x_batch_all[min_idx]

        # print('type', class_prototypes.dtype)
        return torch.tensor(class_prototypes).float()  # todo

    def rand_selection(self, x_batch):
        x_batch = x_batch.numpy()

        if len(x_batch) < self.k:
            r = random.choices(range(0, len(x_batch)), k=self.k)
        else:
            r = random.sample(range(0, len(x_batch)), k=self.k)

        class_prototypes = x_batch[r]
        return torch.tensor(class_prototypes).float()

    def add_prototypes(self, class_prototypes):
        class_prototypes = class_prototypes.unsqueeze(0)
        self.raw_prototypes = torch.cat((self.raw_prototypes, class_prototypes)).to(self.device)

    def extract(self, x_batch):
        self.extractor.eval()
        with torch.no_grad():
            return self.extractor(x_batch.to(self.device))


class IncrementalCentroids(ContinualLearner, ContinualExtractor):

    def __init__(self, extractor: nn.Module, centroids: Tensor, k: int, extractor_lr: float, centroids_lr: float, device='cpu'):
        super().__init__()
        self.extractor = extractor.to(device)
        self.centroids = nn.Parameter(centroids, requires_grad=True).to(device)
        self.extractor_optimizer = Adam(self.extractor.parameters(), lr=extractor_lr) \
            if len(list(self.extractor.parameters())) else None
        self.centroids_optimizer = Adam([self.centroids], lr=centroids_lr) \
            if len(list(self.centroids)) else None
        self.extractor_lr = extractor_lr
        self.centroids_lr = centroids_lr
        self.k = k
        self.device = device

    def predict(self, x_batch):
        x_batch = x_batch.to(self.device)
        preds = torch.max(self.predict_prob(x_batch), 1)[1]

        return preds

    def predict_prob(self, x_batch):
        x_batch = x_batch.to(self.device)
        self.extractor.eval()

        with torch.no_grad():
            centroids_features = self.extractor(torch.flatten(self.centroids, start_dim=0, end_dim=1))
            dists = self.centroids_dists(x_batch, centroids_features)
            min_dists = torch.min(dists, -1).values
            # min_dists_smax = torch.softmax(-StreamingCentroids.norm_min_dists(min_dists, dim=1), dim=1)

        return -min_dists.cpu()

    def centroids_dists(self, x_batch, centroids_features):
        return self.calc_centroids_dists(
            self.extractor(x_batch),
            centroids_features
        )

    def calc_centroids_dists(self, x_batch, centroids_features):
        n, c, k, d = len(x_batch), len(self.centroids), self.k, len(x_batch[0])

        features = x_batch.unsqueeze(1).unsqueeze(1).expand(n, c, k, d)
        centroids_features = centroids_features.reshape(c, k, d).unsqueeze(0).expand(n, c, k, d)
        dists = torch.pow(features - centroids_features, 2).sum(-1)  # [bs, c, k]

        return dists

    def update(self, x_batch, y_batch, **kwargs):
        self.update_model(IndexDataset(TensorDataset(x_batch, y_batch)))

    def update_model(self, data):
        # if len(self.centroids) > 0:
        #     prev_centroids = torch.clone(self.extractor(torch.flatten(self.centroids, start_dim=0, end_dim=1))).detach()

        for _ in range(10):
            self.extractor.train()
            for xb, yb, ib in DataLoader(data, batch_size=128, shuffle=True):
                if len(xb) <= 1:
                    continue
                xb = xb.to(self.device)
                yb = yb.long().to(self.device)

                with torch.no_grad():
                    if self.ensure_class_centroids(xb, yb):
                        self.centroids_optimizer = Adam([self.centroids], lr=self.centroids_lr)

                centroids_features = self.extractor(torch.flatten(self.centroids, start_dim=0, end_dim=1))
                min_dists_norm = self.min_dists(xb, centroids_features)
                loss = IncrementalCentroids.base_centroids_loss(min_dists_norm, yb)

                # todo: try with more centroids and use all in loss
                # if len(self.centroids) > 1:
                #     # stab_loss = 100 * IncrementalCentroids2.stability_loss(centroids_features, prev_centroids)
                #     # print(loss, stab_loss)
                #     # loss += stab_loss
                #     loss += IncrementalCentroids2.contrastive_loss(min_dists_norm, yb)

                loss.backward()

                with torch.no_grad():
                    self.extractor_optimizer.step()
                    self.extractor_optimizer.zero_grad()
                    self.centroids_optimizer.step()
                    self.centroids_optimizer.zero_grad()

    def min_dists(self, x_batch, centroids_features):
        dists = self.centroids_dists(x_batch, centroids_features)
        min_dists = torch.min(dists, -1).values  # [bs, c]

        return IncrementalCentroids.norm_min_dists(min_dists, dim=0)

    @staticmethod
    def norm_min_dists(min_dists, dim):
        return min_dists / torch.max(min_dists, dim=dim).values.detach()  # todo: detach is very important here

    @staticmethod
    def base_centroids_loss(min_dists, y_batch):
        return torch.mean(min_dists.gather(1, y_batch.unsqueeze(1)))

    @staticmethod
    def stability_loss(new_centroids, prev_centroids):
        euc_dists = torch.pow(new_centroids[:len(prev_centroids)] - prev_centroids, 2).sum(-1)  # todo: normalize it?
        # print(euc_dists)
        return torch.mean(euc_dists)

    @staticmethod
    def contrastive_loss(min_dists, y_batch):
        neg_clusters_masks = (1 - one_hot(y_batch, min_dists.shape[1])).bool()  # todo: is this differ? contrastive loss shouldn't update old centroids
        return -torch.mean(min_dists.masked_select(neg_clusters_masks))

    def ensure_class_centroids(self, x_batch, y_batch):
        y_max = int(torch.max(y_batch).item())
        centroids_num = len(self.centroids)
        diff = y_max + 1 - centroids_num
        if diff == 0: return False

        new_centroids = []
        for i in range(diff):
            cls_indices_mask = y_batch == centroids_num + i
            cls_indices = cls_indices_mask.nonzero(as_tuple=True)[0]

            if len(cls_indices) > 0:
                if len(cls_indices) < self.k:
                    r = cls_indices[random.choices(range(0, len(cls_indices)), k=self.k)]
                else:
                    r = cls_indices[random.sample(range(0, len(cls_indices)), k=self.k)]

                class_centroids = torch.clone(x_batch[r]).detach()
                new_centroids.append(class_centroids)
            else:
                new_centroids.append(torch.rand((self.k,) + x_batch.shape[0]))  # todo: poor init if classes do not come in order

        new_centroids = torch.cat((self.centroids.data.detach(), torch.stack(new_centroids)))

        self.centroids = nn.Parameter(new_centroids, requires_grad=True).to(self.device)
        return True

    def extract(self, x_batch):
        self.extractor.eval()
        with torch.no_grad():
            return self.extractor(x_batch.to(self.device))


# class StreamingCentroids(ContinualLearner, ContinualExtractor):
#
#     def __init__(self, extractor: nn.Module, centroids: Tensor, k: int, extractor_lr: float, centroids_lr: float, device='cpu'):
#         super().__init__()
#         self.extractor = extractor.to(device)
#         self.centroids = nn.Parameter(centroids, requires_grad=True).to(device)
#         self.extractor_optimizer = Adam(self.extractor.parameters(), lr=extractor_lr) \
#             if len(list(self.extractor.parameters())) else None
#         self.centroids_optimizer = Adam([self.centroids], lr=centroids_lr) \
#             if len(list(self.centroids)) else None
#         self.extractor_lr = extractor_lr
#         self.centroids_lr = centroids_lr
#         self.k = k
#         self.device = device
#
#     def predict(self, x_batch):
#         x_batch = x_batch.to(self.device)
#         preds = torch.max(self.predict_prob(x_batch), 1)[1]
#
#         return preds
#
#     def predict_prob(self, x_batch):
#         x_batch = x_batch.to(self.device)
#         self.extractor.eval()
#
#         with torch.no_grad():
#             dists = self.centroids_dists(x_batch)
#             min_dists = torch.min(dists, -1).values
#             # min_dists_smax = torch.softmax(-StreamingCentroids.norm_min_dists(min_dists, dim=1), dim=1)
#
#         return -min_dists.cpu()
#
#     def centroids_dists(self, x_batch):
#         return self.calc_centroids_dists(
#             self.extractor(x_batch),
#             self.extractor(torch.flatten(self.centroids, start_dim=0, end_dim=1))
#         )
#
#     def calc_centroids_dists(self, x_batch, centroids_features):
#         n, c, k, d = len(x_batch), len(self.centroids), self.k, len(x_batch[0])
#
#         features = x_batch.unsqueeze(1).unsqueeze(1).expand(n, c, k, d)
#         centroids_features = centroids_features.reshape(c, k, d).unsqueeze(0).expand(n, c, k, d)
#         dists = torch.pow(features - centroids_features, 2).sum(-1)  # [bs, c, k]
#
#         return dists
#
#     def update(self, x_batch, y_batch, **kwargs):
#         x_batch = x_batch.to(self.device)
#         y_batch = y_batch.long().to(self.device)
#         if self.ensure_class_centroids(x_batch, y_batch):
#             self.centroids_optimizer = Adam([self.centroids], lr=self.centroids_lr)
#
#         self.extractor.train()
#         min_dists_norm = self.min_dists(x_batch)
#         loss = StreamingCentroids.base_centroids_loss(min_dists_norm, y_batch)
#         # print(min_dists.shape, loss)
#         loss.backward()
#
#         with torch.no_grad():
#             self.extractor_optimizer.step()
#             self.extractor_optimizer.zero_grad()
#             self.centroids_optimizer.step()
#             self.centroids_optimizer.zero_grad()
#
#         # post-hoc
#         # min_dists_norm = self.min_dists(x_batch)
#         # post_loss = StreamingCentroids.base_centroids_loss(min_dists_norm, y_batch)
#         # post_loss.backward()
#         #
#         # with torch.no_grad():
#         #     self.centroids_optimizer.step()
#         #     self.centroids_optimizer.zero_grad()
#
#     def min_dists(self, x_batch):
#         dists = self.centroids_dists(x_batch)
#         min_dists = torch.min(dists, -1).values  # [bs, c]
#
#         return StreamingCentroids.norm_min_dists(min_dists, dim=0)
#
#     @staticmethod
#     def norm_min_dists(min_dists, dim):
#         return min_dists / torch.max(min_dists, dim=dim).values.detach()  # todo: detach is very important here
#
#     @staticmethod
#     def base_centroids_loss(min_dists, y_batch):
#         return torch.mean(min_dists.gather(1, y_batch.unsqueeze(1)))
#
#     @staticmethod
#     def contrastive_loss(min_dists, y_batch):
#         neg_clusters_masks = (1 - one_hot(y_batch, min_dists.shape[1])).bool()  # todo: is this differ?
#         return -torch.mean(min_dists.masked_select(neg_clusters_masks))
#
#     def ensure_class_centroids(self, x_batch, y_batch):
#         y_max = int(torch.max(y_batch).item())
#         centroids_num = len(self.centroids)
#         diff = y_max + 1 - centroids_num
#         if diff == 0: return False
#
#         new_centroids = []
#         for i in range(diff):
#             cls_indices_mask = y_batch == centroids_num + i
#             cls_indices = cls_indices_mask.nonzero(as_tuple=True)[0]
#
#             if len(cls_indices) > 0:
#                 if len(cls_indices) < self.k:
#                     r = cls_indices[random.choices(range(0, len(cls_indices)), k=self.k)]
#                 else:
#                     r = cls_indices[random.sample(range(0, len(cls_indices)), k=self.k)]
#
#                 class_centroids = torch.clone(x_batch[r]).detach()
#                 new_centroids.append(class_centroids)
#             else:
#                 new_centroids.append(torch.rand((self.k,) + x_batch[0]))  # todo: poor init if classes do not come in order
#
#         new_centroids = torch.vstack((self.centroids.data.detach(), torch.stack(new_centroids))) if len(self.centroids) > 0 \
#             else torch.stack(new_centroids)
#
#         self.centroids = nn.Parameter(new_centroids, requires_grad=True).to(self.device)  # todo: to(device)?
#         return True
#
#     def extract(self, x_batch):
#         self.extractor.eval()
#         with torch.no_grad():
#             return self.extractor(x_batch.to(self.device))


