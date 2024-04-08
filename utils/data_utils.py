import torch
import random


class DataUtils:

    @staticmethod
    def rand_selection_with_idx(x_batch, n, device='cpu'):
        if len(x_batch) < n:
            r = torch.tensor(random.choices(range(0, len(x_batch)), k=n)).to(device)
        else:
            r = torch.tensor(random.sample(range(0, len(x_batch)), k=n)).to(device)

        samples = torch.index_select(x_batch, 0, r)

        return samples, r

    @staticmethod
    def rand_selection(x_batch, n, device='cpu'):
        samples, _ = DataUtils.rand_selection_with_idx(x_batch, n, device)
        return samples

    @staticmethod
    def per_component_selection(x_batch, assignments, n, k, loss_type, device='cpu'):
        nk = int(n / k)
        component_samples = []

        for i in range(k):
            i_x = x_batch[assignments == i]
            component_samples.append(DataUtils.rand_selection(i_x, nk, device))

        samples = torch.cat(component_samples) if loss_type != 'mpr' else torch.stack(component_samples)

        return samples

