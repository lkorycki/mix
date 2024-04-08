import torch
import avalanche as ava
from avalanche.models import DynamicModule
from avalanche.training import BaseStrategy, ICaRLLossPlugin

from core.clearn import AvalancheContinualLearner

from learners.nnet import ConvNeuralNet


class AvaLearner(AvalancheContinualLearner):

    def __init__(self, strategy: BaseStrategy, device='cpu'):
        super().__init__()
        self.strategy = strategy
        self.device = device

    @torch.no_grad()
    def predict(self, x_batch):
        probs = self.predict_prob(x_batch)
        preds = torch.max(probs, dim=-1)[1].cpu()
        return preds

    @torch.no_grad()
    def predict_prob(self, x_batch):
        model = self.strategy.model
        model.eval()
        return model(x_batch.to(self.device))

    def update(self, experience, **kwargs):
        self.strategy.train(experience, eval=[])


class ICaRL(AvaLearner):

    def __init__(self, model: ConvNeuralNet, memory_size, optimizer, scheduler, epochs, batch_size, device='cpu'):
        super().__init__(
            strategy=ava.training.ICaRL(
                feature_extractor=model.extractor,
                classifier=model.classifier,
                optimizer=optimizer,
                memory_size=memory_size,
                buffer_transform=None,
                fixed_memory=False,
                train_mb_size=batch_size,
                train_epochs=epochs,
                plugins=[scheduler] if scheduler is not None else None,
                criterion=ICaRLLossPlugin(),
                device=device
            ),
            device=device
        )

    @torch.no_grad()
    def predict_prob(self, x_batch):
        model: DynamicModule = self.strategy.model
        model.eval().adaptation()
        return model(x_batch.to(self.device))


class AGEM(AvaLearner):

    def __init__(self, model: ConvNeuralNet, memory_size, sample_size, criterion, optimizer, scheduler, epochs, batch_size,
                 device='cpu'):
        super().__init__(
            strategy=ava.training.AGEM(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                patterns_per_exp=memory_size,
                sample_size=sample_size,
                train_mb_size=batch_size,
                train_epochs=epochs,
                device=device,
                plugins=[scheduler] if scheduler is not None else None
            ),
            device=device
        )


class GSSGreedy(AvaLearner):

    def __init__(self, model: ConvNeuralNet, input_size, memory_size, criterion, optimizer, scheduler, epochs, batch_size,
                 device='cpu'):
        super().__init__(
            strategy=ava.training.GSS_greedy(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                input_size=input_size,
                mem_size=memory_size,
                train_mb_size=batch_size,
                train_epochs=epochs,
                device=device,
                plugins=[scheduler] if scheduler is not None else None
            ),
            device=device
        )


class SI(AvaLearner):

    def __init__(self, model: ConvNeuralNet, si_lambda, criterion, optimizer, scheduler, epochs, batch_size,
                 device='cpu'):
        super().__init__(
            strategy=ava.training.SynapticIntelligence(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                si_lambda=si_lambda,
                train_mb_size=batch_size,
                train_epochs=epochs,
                device=device,
                plugins=[scheduler] if scheduler is not None else None
            ),
            device=device
        )


class LWF(AvaLearner):

    def __init__(self, model: ConvNeuralNet, alpha, temperature, criterion, optimizer, scheduler, epochs, batch_size,
                 device='cpu'):
        super().__init__(
            strategy=ava.training.LwF(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                alpha=alpha,
                temperature=temperature,
                train_mb_size=batch_size,
                train_epochs=epochs,
                device=device,
                plugins=[scheduler] if scheduler is not None else None
            ),
            device=device
        )
