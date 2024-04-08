from abc import ABC, abstractmethod


class GenericContinualLearner(ABC):
    @abstractmethod
    def predict(self, x_batch):
        pass

    @abstractmethod
    def predict_prob(self, x_batch):
        pass


class ContinualLearner(GenericContinualLearner):

    def __init__(self):
        pass

    def initialize(self, x_batch, y_batch, **kwargs):
        self.update(x_batch, y_batch, **kwargs)

    @abstractmethod
    def update(self, x_batch, y_batch, **kwargs):
        pass


class AvalancheContinualLearner(GenericContinualLearner):

    def __init__(self):
        pass

    @abstractmethod
    def update(self, experience, **kwargs):
        pass


class ContinualExtractor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def extract(self, x_batch):
        pass


class LossTracker(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass


class TimeTracker:

    def __init__(self):
        self.time_metrics = {'update': [0, 0.0], 'prediction': [0, 0.0]}

    def get_time_metrics(self):
        return {
            'update': self.time_metrics['update'][1] / self.time_metrics['update'][0],
            'prediction': self.time_metrics['prediction'][1] / self.time_metrics['prediction'][0]
        } if self.time_metrics['update'][0] > 0 and self.time_metrics['prediction'][0] > 0 else None

    def reset_time_metrics(self):
        self.time_metrics = {'update': [0, 0.0], 'prediction': [0, 0.0]}
