import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from core.clearn import ContinualLearner
import torch.nn.functional as F
import numpy as np

from learners.wrappers.mth_utils import MammothBuffer

'''
Base code source thanks to: https://github.com/aimagelab/mammoth
Papers/configuration: 
- DER (https://arxiv.org/abs/2004.07211)
- AGEM (https://arxiv.org/abs/1812.00420)
'''


class MammothLearner(ContinualLearner):

    def __init__(self, device='cpu'):
        super().__init__()
        self.model: ContinualModel = None
        self.device = device

    def predict(self, x_batch):
        probs = self.predict_prob(x_batch)
        preds = torch.max(probs, dim=-1)[1].cpu()

        return preds

    def predict_prob(self, x_batch):
        out = self.model(x_batch.to(self.device))
        probs = F.softmax(out, dim=1)

        return probs

    def update(self, x_batch, y_batch, **kwargs):
        self.model.observe(x_batch, y_batch, x_batch)


class DER(MammothLearner):

    def __init__(self, model, buffer_size, minibatch_size, alpha, loss, optimizer, scheduler, buffer_mode='reservoir',
                 num_classes=-1, device='cpu'):
        super().__init__(device)
        self.model = _DER(model.to(device), buffer_size, minibatch_size, alpha, loss, optimizer, buffer_mode, num_classes,
                          device=device)


class AGEM(MammothLearner):

    def __init__(self, model, buffer_size, minibatch_size, loss, optimizer, scheduler, batch_size, buffer_mode,
                 num_classes=-1, device='cpu'):
        super().__init__(device)
        self.model = _AGEM(model.to(device), buffer_size, minibatch_size, loss, optimizer, buffer_mode, num_classes,
                           device=device)
        self.batch_size = batch_size
        self.device = device

    def update(self, x_class_batch, y_class_batch, **kwargs):
        class_batch = TensorDataset(x_class_batch, y_class_batch)

        for xb, yb in DataLoader(class_batch, batch_size=self.batch_size, shuffle=True):
            self.model.observe(xb.to(self.device), yb.long().to(self.device), xb)

        self.model.end_task(x_class_batch, y_class_batch.long())


class ContinualModel(nn.Module):

    def __init__(self, model, loss, optimizer, transform=None, device='cpu'):
        super(ContinualModel, self).__init__()
        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.transform = transform
        self.device = device

    def forward(self, x):
        return self.model(x)

    def observe(self, inputs, labels, not_aug_inputs):
        pass


class _DER(ContinualModel):

    def __init__(self, model, buffer_size, minibatch_size, alpha, loss, optimizer, buffer_mode='reservoir',
                 num_classes=-1, transform=None, device='cpu'):
        super(_DER, self).__init__(model.to(device), loss, optimizer, transform, device)
        self.buffer_size = buffer_size
        self.buffer = MammothBuffer(self.buffer_size, buffer_mode, num_classes, self.device)
        self.minibatch_size = minibatch_size
        self.alpha = alpha

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        outputs = self.model(inputs.to(self.device))
        loss = self.loss(outputs, labels.long().to(self.device))

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.minibatch_size, transform=self.transform)
            buf_outputs = self.model(buf_inputs)
            loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)


class _AGEM(ContinualModel):

    def __init__(self, model, buffer_size, minibatch_size, loss, optimizer, buffer_mode='reservoir',
                 num_classes=-1, transform=None, device='cpu'):
        super(_AGEM, self).__init__(model, loss, optimizer, transform, device)
        self.buffer_size = buffer_size
        self.buffer = MammothBuffer(self.buffer_size, buffer_mode, num_classes, self.device)
        self.minibatch_size = minibatch_size

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, x_class_batch, y_class_batch):
        self.buffer.add_data(
            examples=x_class_batch,
            labels=y_class_batch
        )

    def observe(self, inputs, labels, not_aug_inputs):
        self.zero_grad()
        p = self.model.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if not self.buffer.is_empty():
            self.store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.minibatch_size, transform=self.transform)
            self.model.zero_grad()
            buf_outputs = self.model.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            self.store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = self.project(gxy=self.grad_xy, ger=self.grad_er)
                self.overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                self.overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()

    @staticmethod
    def store_grad(params, grads, grad_dims):
        grads.fill_(0.0)
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = np.sum(grad_dims[:count + 1])
                grads[begin: end].copy_(param.grad.data.view(-1))
            count += 1

    @staticmethod
    def overwrite_grad(params, newgrad, grad_dims):
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1

    @staticmethod
    def project(gxy, ger):
        corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
        return gxy - corr * ger


