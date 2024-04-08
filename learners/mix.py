from math import ceil

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset

from core.clearn import ContinualLearner, ContinualExtractor, LossTracker, TimeTracker
from torch import nn
import torch
import torch.nn.functional as F
from utils.clust_utils import ClustUtils as cu
from data.data_utils import IndexDataset
from timeit import default_timer as timer
from utils.data_utils import DataUtils as du


class MIX(ContinualLearner, ContinualExtractor, LossTracker, TimeTracker):

    def __init__(self, k, extractor, init_method, replay_buffer_size, comp_select, loss_type, inter_tightness,
                 intra_tightness, use_annealing, sharp_annealing, full_cov, cov_min, classification_method,
                 extractor_lr, gmm_lr, epochs, batch_size, disable_inter_contrast=False, super_batch_classes=8,
                 replay_buffer_device='cpu', device='cpu'):
        TimeTracker.__init__(self)
        self.k = k
        self.model: DeepGMM = DeepGMM(extractor).to(device)
        self.model.eval()
        self.region_cache = {}
        self.replay_buffer = {}
        self.replay_buffer_size = replay_buffer_size
        self.comp_select = comp_select
        self.loss_type = loss_type
        self.inter_t = inter_tightness
        self.intra_t = intra_tightness
        self.init_method = init_method
        self.use_annealing = use_annealing
        self.sharp_annealing = sharp_annealing
        self.full_cov = full_cov
        self.classification_method = classification_method
        self.extractor_optimizer = None
        self.extractor_scheduler = None
        self.extractor_lr = extractor_lr
        self.gmm_optimizer = None
        self.gmm_lr = gmm_lr
        self.epochs = epochs
        self.cov_min = cov_min
        self.loss = 0.0
        self.batch_size = batch_size
        self.inter_contrast = not disable_inter_contrast
        self.fixed_extractor = isinstance(self.model.extractor, nn.Identity)
        self.super_batch_size = super_batch_classes * self.batch_size
        self.replay_buffer_device = replay_buffer_device
        self.device = device

        if self.loss_type == 'mpr' and not self.comp_select:
            raise ValueError('Loss type: mpr requires comp_select=True!')
        elif self.comp_select and self.init_method != 'k_means':
            raise ValueError('Per component selection (comp_select=True) requires k-means initialization!')

    def predict(self, x_batch):
        start = timer()

        probs = self.predict_prob(x_batch)
        preds = torch.max(probs, dim=-1)[1].cpu()

        self.time_metrics['prediction'][1] += timer() - start
        self.time_metrics['prediction'][0] += 1

        return preds

    def predict_prob(self, x_batch):
        with torch.no_grad():
            out = self.model(x_batch.to(self.device))

            if self.classification_method == 'max_component':
                probs = torch.max(out, dim=-1)[0]
            elif self.classification_method == 'softmax':
                probs = torch.softmax(torch.sum(out, dim=-1), dim=1)
            else:
                raise ValueError('Classification method: {0} is not supported!'.format(self.classification_method))

            return probs

    def update(self, x_class_batch, y_class_batch, **kwargs):
        start = timer()

        class_batch = IndexDataset(TensorDataset(x_class_batch, y_class_batch))
        new_cls = int(y_class_batch[0])
        epoch = kwargs.get('epoch')

        if len(self.model.mixtures) < new_cls + 1:
            self.__init_new_class(new_cls, class_batch)
        elif new_cls == 0 and epoch > 0:
            return

        losses = []
        self.__set_model_train()

        for xb, _, ib in DataLoader(class_batch, batch_size=self.batch_size, shuffle=True):
            if len(xb) <= 1:
                continue

            self.__zero_optimizer(self.extractor_optimizer)
            self.__zero_optimizer(self.gmm_optimizer)

            loss = self.__loss_type(new_cls, xb, ib, epoch)
            losses.append(loss)

            with torch.no_grad():
                self.__step_optimizer(self.extractor_optimizer)
                self.__step_optimizer(self.gmm_optimizer)
                self.model.clamp_covs(self.cov_min)

        if (not self.fixed_extractor) and self.use_annealing:
            self.extractor_scheduler.step()

        self.model.eval()
        self.loss = sum(losses) / len(losses)

        self.time_metrics['update'][1] += timer() - start
        self.time_metrics['update'][0] += 1

    @staticmethod
    def __zero_optimizer(optimizer):
        if optimizer is not None:
            optimizer.zero_grad()

    @staticmethod
    def __step_optimizer(optimizer):
        if optimizer is not None:
            optimizer.step()

    def __init_new_class(self, new_cls, class_batch):
        x_class_batch, xf_class_batch, ibs = self.__prepare_features(class_batch)
        component_assignments = self.__init_mixture(xf_class_batch)

        if self.replay_buffer_size > 0:
            self.replay_buffer[new_cls] = self.__prepare_replay_samples(x_class_batch, component_assignments).to(
                self.replay_buffer_device)

        if self.loss_type == 'mpr':
            self.__update_region_cache(component_assignments, ibs)

        self.__update_optimizers()

    def __prepare_features(self, class_batch):
        x_class_batch, xf_class_batch, ibs = [], [], []

        for xb, _, ib in DataLoader(class_batch, batch_size=self.batch_size, shuffle=False):
            xf_class_batch.append(self.extract(xb.to(self.device)).cpu())
            x_class_batch.append(xb)
            ibs.append(ib)

        return torch.cat(x_class_batch), torch.cat(xf_class_batch), torch.cat(ibs)

    def extract(self, x_batch):
        with torch.no_grad():
            return self.model.extractor(x_batch.to(self.device))

    def __init_mixture(self, xf_class_batch):
        means, covs, assignments = self.__init_method(xf_class_batch, self.init_method, self.full_cov)
        weights = torch.ones(self.k) * (1.0 / self.k)

        self.model.add_mixture(means, covs, weights, self.device)
        self.model.clamp_covs(self.cov_min)

        return assignments

    def __init_method(self, xf_class_batch, strategy, full_cov):
        if strategy == 'k_means':
            return self.__k_means_init(xf_class_batch, full_cov, self.comp_select)
        elif strategy == 'heuristic':
            return self.__heuristic_init(xf_class_batch, full_cov)
        elif strategy == 'rand':
            return self.__rand_init(xf_class_batch, full_cov)
        else:
            raise ValueError('Init method: {0} is not supported!'.format(self.init_method))

    def __rand_init(self, x, full_cov):
        means = du.rand_selection(x, self.k)
        d = len(x[0])

        if full_cov:
            covs = (torch.ones(d, d) * self.cov_min).repeat(self.k, 1, 1)
        else:
            covs = (torch.ones(d) * self.cov_min).repeat(self.k, 1)

        return means, covs, None

    def __heuristic_init(self, x_batch, full_cov):
        means_indices = {0}

        for i in range(self.k - 1):
            max_min_dist, max_min_idx = -float("inf"), -1

            for j in range(len(x_batch)):
                if j in means_indices:
                    continue

                min_dist = float("inf")
                for xm_idx in means_indices:
                    dist = ((x_batch[xm_idx] - x_batch[j]) ** 2).sum(axis=0)
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    max_min_idx = j

            means_indices.add(max_min_idx)

        means = torch.index_select(x_batch, 0, torch.tensor(list(means_indices)))

        if full_cov:
            covs = torch.cov(x_batch.T).repeat(self.k, 1, 1) / self.k
        else:
            covs = torch.var(x_batch, dim=0).repeat(self.k, 1) / self.k

        return means, covs, None

    def __k_means_init(self, x_batch, full_cov, comp_select):
        assignments, means = cu.k_means(x_batch, self.k)

        if len(torch.unique(assignments)) != self.k:
            means, covs, _ = self.__broken_extractor_handler(x_batch)
            assignments = torch.randint(0, self.k, (len(assignments),))
        else:
            covs = []
            for i in range(self.k):
                i_x = x_batch[assignments == i]
                cov = torch.cov(i_x.T) if full_cov else torch.var(x_batch, dim=0)
                covs.append(cov)

            covs = torch.stack(covs)

        return means, covs, assignments if comp_select else None

    def __broken_extractor_handler(self, x_batch):
        return self.__rand_init(x_batch, self.full_cov)

    def __prepare_replay_samples(self, x_class_batch, assignments):
        if assignments is not None:
            return du.per_component_selection(x_class_batch, assignments, self.replay_buffer_size, self.k,
                                              self.loss_type, 'cpu')
        else:
            return du.rand_selection(x_class_batch, self.replay_buffer_size, 'cpu')

    def __update_region_cache(self, assignments, ibs):
        self.region_cache = {}

        for i in range(self.k):
            i_ibs = ibs[assignments == i]
            self.region_cache.update({idx.item(): i for idx in i_ibs})

    def __update_optimizers(self):
        if not self.fixed_extractor:
            self.extractor_optimizer = Adam(self.model.extractor.parameters(), lr=self.extractor_lr)

            if self.use_annealing:
                if self.sharp_annealing:
                    self.extractor_scheduler = torch.optim.lr_scheduler.StepLR(self.extractor_optimizer,
                                                                               step_size=self.epochs - 1, gamma=0.0)
                else:
                    lf = lambda x: 1.0 - 10.0 ** (-2.0 * (1.0 - x / (self.epochs - 1.0)))
                    self.extractor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.extractor_optimizer, lr_lambda=lf)

        self.gmm_optimizer = Adam(self.model.mixtures.parameters(), lr=self.gmm_lr)

    def __set_model_train(self):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __loss_type(self, y, xb, ib, epoch):
        if self.loss_type == 'ce':
            return self.__ce_update(y, xb)
        elif self.loss_type == 'mp':
            return self.__mp_update(y, xb, epoch)
        elif self.loss_type == 'mpr':
            return self.__mpr_update(y, xb, ib, epoch)
        else:
            raise ValueError('Loss type: {0} is not supported!'.format(self.loss_type))

    def __ce_update(self, new_cls, xb):
        total_loss = 0.0
        logits = torch.sum(self.model(xb.to(self.device)), dim=-1)

        y_target = torch.ones(len(xb)).to(self.device).long() * new_cls
        loss = F.cross_entropy(logits, y_target)
        loss.backward()
        total_loss += loss.item()

        replay_buffer = {k: v for k, v in self.replay_buffer.items() if k != new_cls}

        if len(replay_buffer) > 0:
            sb_num = ceil((len(replay_buffer) * self.batch_size) / self.super_batch_size)
            sb_cls_num = int(self.super_batch_size / self.batch_size)
            class_buffers = list(replay_buffer.items())

            for i in range(sb_num):
                rxbs, ys = self.__form_super_batch(class_buffers, i, sb_cls_num)
                rxbs_logits = torch.sum(self.model(rxbs), dim=-1)
                loss = 0.0

                for j, y in enumerate(ys):
                    logits = rxbs_logits[j * self.batch_size:(j + 1) * self.batch_size]
                    y_target = torch.ones(len(logits)).to(self.device).long() * y
                    loss += F.cross_entropy(logits, y_target)

                loss.backward()
                total_loss += loss.item()

        return total_loss

    def __mp_update(self, new_cls, xb, epoch):
        total_loss = 0.0
        na = self.neg_loss_annealing(epoch, self.epochs, self.sharp_annealing) if self.use_annealing else 1.0

        probs = self.model(xb.to(self.device))
        loss = self.__single_class_loss(new_cls, probs, self.inter_contrast and (new_cls > 0), na)
        loss.backward()
        total_loss += loss.item()

        replay_buffer = {k: v for k, v in self.replay_buffer.items() if k != new_cls}

        if len(replay_buffer) > 0:
            sb_num = ceil((len(replay_buffer) * self.batch_size) / self.super_batch_size)
            sb_cls_num = int(self.super_batch_size / self.batch_size)
            class_buffers = list(replay_buffer.items())

            for i in range(sb_num):
                rxbs, ys = self.__form_super_batch(class_buffers, i, sb_cls_num)
                rxbs_probs = self.model(rxbs)
                loss = 0.0

                for j, y in enumerate(ys):
                    probs = rxbs_probs[j * self.batch_size:(j + 1) * self.batch_size]
                    loss += self.__single_class_loss(y, probs, self.inter_contrast, na)

                loss.backward()
                total_loss += loss.item()

        return total_loss

    def __single_class_loss(self, y, probs, inter_contrast, na):
        max_probs, _ = torch.max(probs, dim=-1)
        max_probs_mean = torch.mean(max_probs, dim=0)
        loss = -max_probs_mean[y]

        if inter_contrast:
            max_probs_mean[y] = self.tau(self.inter_t, self.get_component_max_avg(y))
            loss += na * torch.max(max_probs_mean)

        return loss

    def tau(self, t, mx):
        return (-1.0 / t) + mx

    def __form_super_batch(self, class_buffers, i, sb_cls_num):
        y_rxbs = class_buffers[i * sb_cls_num:min((i+1) * sb_cls_num, len(class_buffers))]
        ys, rxbs = [], []

        for y, rxb in y_rxbs:
            ys.append(y)
            rxbs.append(du.rand_selection(rxb.to(self.device), self.batch_size, self.device))

        return torch.cat(rxbs), ys

    def __mpr_update(self, new_cls, xb, ib, epoch):
        total_loss, loss = 0.0, 0.0
        na = self.neg_loss_annealing(epoch, self.epochs - 1, self.sharp_annealing) if self.use_annealing else 1.0
        assignments = torch.tensor([self.region_cache[idx.item()] for idx in ib])

        probs = self.model(xb.to(self.device))

        for m in range(self.k):
            m_probs = probs[assignments == m]
            loss += self.__single_class_component_loss(new_cls, m, m_probs, self.inter_contrast and (new_cls > 0),
                                                       self.k > 1, na)

        loss.backward()
        total_loss += loss.item()

        replay_buffer = {k: v for k, v in self.replay_buffer.items() if k != new_cls}

        if len(replay_buffer) > 0:
            kbs = int(self.batch_size / self.k)
            sb_num = ceil((len(replay_buffer) * self.batch_size) / self.super_batch_size)
            sb_cls_num = int(self.super_batch_size / self.batch_size)
            class_buffers = list(replay_buffer.items())

            for i in range(sb_num):
                rxbs, ys = self.__form_regions_super_batch(class_buffers, i, sb_cls_num, kbs)
                rxbs_probs = self.model(rxbs)
                loss = 0.0

                for j, y in enumerate(ys):
                    for m in range(self.k):
                        probs = rxbs_probs[j * self.batch_size + m * kbs:j * self.batch_size + (m + 1) * kbs]
                        loss += self.__single_class_component_loss(y, m, probs, self.inter_contrast, self.k > 1, na)

                loss.backward()
                total_loss += loss.item()

        return total_loss

    def __single_class_component_loss(self, y, m, probs, inter_contrast, intra_contrast, na):
        n = int(inter_contrast) + int(intra_contrast)
        component_probs = probs[:, y, m]
        loss = -torch.mean(component_probs)

        if inter_contrast:
            max_probs, _ = torch.max(probs, dim=-1)
            max_probs = torch.mean(max_probs, dim=0)
            max_probs[y] = self.tau(self.inter_t, self.get_component_max(y, m))
            loss += na * torch.max(max_probs) / n

        if intra_contrast:
            y_mean_probs = torch.mean(probs[:, y], dim=0)
            y_mean_probs[m] = self.tau(self.intra_t, self.get_component_max(y, m))
            loss += na * torch.max(y_mean_probs) / n

        return loss

    def __form_regions_super_batch(self, class_buffers, i, sb_cls_num, kbs):
        y_rxbs = class_buffers[i * sb_cls_num:min((i+1) * sb_cls_num, len(class_buffers))]

        ys, rxbs = [], []
        for y, rxb in y_rxbs:
            ys.append(y)
            rxb = rxb.to(self.device)

            for m in range(self.k):
                rxbs.append(du.rand_selection(rxb[m], kbs, self.device))

        return torch.cat(rxbs), ys

    @staticmethod
    def neg_loss_annealing(epoch, max_epoch, sharp):
        if sharp:
            return 0.0 if epoch == max_epoch else 1.0
        else:
            last_epoch = max_epoch - 2.0
            return 1.0 - 10.0 ** (-2.0 * (1.0 - epoch / (last_epoch - 1.0))) if epoch <= last_epoch else 0.0

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.loss

    def get_component_max(self, i, m):
        return self.model.get_mixtures()[i].get_components()[m].get_max()

    def get_component_max_avg(self, i):
        component_maxs = [self.model.get_mixtures()[i].get_components()[m].get_max() for m in range(self.k)]
        return sum(component_maxs) / len(component_maxs)


class DeepGMM(nn.Module):

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.mixtures = nn.ModuleList([])

    def forward(self, x_batch):
        x_batch = self.extractor(x_batch)
        return torch.stack([mixture(x_batch) for mixture in self.mixtures]).transpose(0, 1)

    def add_mixture(self, means, covs, weights, device):
        self.mixtures.append(GMM(means, covs, weights).to(device))

    def get_mixture_prob(self, i, x_batch):
        x_batch = self.extractor(x_batch)
        return self.mixtures[i](x_batch)

    def get_mixtures(self):
        return self.mixtures

    def clamp_covs(self, cov_min):
        for mixture in self.mixtures:
            for component in mixture.get_components():
                component.clamp_cov(cov_min)


class GMM(nn.Module):

    def __init__(self, means, covs, weights):
        super().__init__()
        self.components = nn.ModuleList([])
        self.weights = nn.ParameterList([])
        self.weights_normalizer = nn.Softmax(dim=0)

        for mean, cov, weight in zip(means, covs, weights):
            self.add_component(mean, cov, weight)

    def forward(self, x_batch):
        norm_weights = self.weights_normalizer(torch.stack([w for w in self.weights]))
        component_probs = torch.stack([component(x_batch) for component in self.components]).T

        mixture_probs = norm_weights[None, :] * component_probs

        return mixture_probs

    def add_component(self, mean, cov, weight):
        self.components.append(GaussianComponent(mean, cov))
        self.weights.append(nn.Parameter(weight))

    def get_components(self):
        return self.components

    def get_weights(self):
        return self.weights


class GaussianComponent(nn.Module):

    def __init__(self, mean, cov):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.cov = nn.Parameter(cov)
        self.max = -float("inf")

    def forward(self, x_batch):
        if len(self.cov.shape) == 2:
            mvn = torch.distributions.MultivariateNormal(self.mean, scale_tril=torch.tril(self.cov))
        else:
            mvn = torch.distributions.MultivariateNormal(self.mean, torch.diag(self.cov))

        probs = mvn.log_prob(x_batch)
        self.max = max(torch.max(probs).item(), self.max)

        return probs

    def get_statistics(self):
        return self.mean, self.cov

    def clamp_cov(self, cov_min):
        with torch.no_grad():
            self.cov.nan_to_num_(nan=cov_min)

            if len(self.cov.shape) == 2:
                diag = self.cov.diagonal().clone().clamp_(min=cov_min)
                self.cov.clamp_(min=-cov_min)
                self.cov.diagonal().copy_(diag)
            else:
                self.cov.clamp_(min=cov_min)

    def get_max(self):
        return self.max


