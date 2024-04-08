from core.clearn import ContinualLearner
import torch


class OfflinePredictor(ContinualLearner):

    def __init__(self, model, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, x_batch):
        probs = self.predict_prob(x_batch)
        return torch.max(probs, dim=-1)[1].cpu()

    @torch.no_grad()
    def predict_prob(self, x_batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(x_batch.to(self.device)).cpu()

    def update(self, x_batch, y_batch, **kwargs):
        pass
