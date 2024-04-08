import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn as nn

import data.data_collection as data_col
from data.data_labels import get_cls_names
from data.stream import ClassStream
from eval.eval import OfflineClassStreamEvaluator
from eval.experiment import Experiment
from learners.nnet import NeuralNet, CifarResNet18, MnistNet

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ExperimentOfflineExample(Experiment):

    def prepare(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logdir_root = 'runs/offline/ep100'

        self.add_data_creator('MNIST-CI',
                              lambda: ClassStream(data_col.get('MNIST-TRAIN'), data_col.get('MNIST-TEST'), class_size=1))
        self.add_data_creator('FASHION-CI',
                              lambda: ClassStream(data_col.get('FASHION-TRAIN'), data_col.get('FASHION-TEST'),
                                                 class_size=1, cls_names=get_cls_names('FASHION')))
        self.add_data_creator('SVHN-CI',
                              lambda: ClassStream(data_col.get('SVHN-TRAIN'), data_col.get('SVHN-TEST'), class_size=1))
        self.add_data_creator('CIFAR10-CI',
                              lambda: ClassStream(data_col.get('CIFAR10-TRAIN'), data_col.get('CIFAR10-TEST'),
                                                 class_size=1, cls_names=get_cls_names('CIFAR10')))
        self.add_data_creator('CIFAR20C-CI',
                              lambda: ClassStream(data_col.get('CIFAR20C-TRAIN'), data_col.get('CIFAR20C-TEST'),
                                                 class_size=20, class_frac=0.2, cls_names=get_cls_names('CIFAR20C')))
        self.add_data_creator('IMAGENET10-CI',
                              lambda: ClassStream(data_col.get('IMAGENET10-TRAIN'), data_col.get('IMAGENET10-TEST'),
                                                 class_size=1, cls_names=get_cls_names('IMAGENET10')))
        self.add_data_creator('IMAGENET20A-CI',
                              lambda: ClassStream(data_col.get('IMAGENET20A-TRAIN'), data_col.get('IMAGENET20A-TEST'),
                                                 class_size=20, cls_names=get_cls_names('IMAGENET20A')))
        self.add_data_creator('IMAGENET20B-CI',
                              lambda: ClassStream(data_col.get('IMAGENET20B-TRAIN'), data_col.get('IMAGENET20B-TEST'),
                                                 class_size=20, cls_names=get_cls_names('IMAGENET20B')))
        self.add_data_creator('CIFAR100-PRE10-CI',
                              lambda: ClassStream(data_col.get('CIFAR100-PRE10-TRAIN'), data_col.get('CIFAR100-PRE10-TEST'),
                                                 class_size=100, cls_names=get_cls_names('CIFAR100')))
        self.add_data_creator('CIFAR100-PRE100-CI',
                              lambda: ClassStream(data_col.get('CIFAR100-PRE100-TRAIN'), data_col.get('CIFAR100-PRE100-TEST'),
                                                 class_size=100, cls_names=get_cls_names('CIFAR100')))
        self.add_data_creator('IMAGENET200-PRE20B-CI',
                              lambda: ClassStream(data_col.get('IMAGENET200-PRE20B-TRAIN'), data_col.get('IMAGENET200-PRE20B-TEST'),
                                                 class_size=200, cls_names=get_cls_names('IMAGENET200')))
        self.add_data_creator('IMAGENET200-PRE200-CI',
                              lambda: ClassStream(data_col.get('IMAGENET200-PRE200-TRAIN'), data_col.get('IMAGENET200-PRE200-TEST'),
                                                 class_size=200, cls_names=get_cls_names('IMAGENET200')))

        def mnist_net_creator():
            mnist_net = MnistNet(in_size=(1, 28, 28), out_size=10)
            optimizer = SGD(mnist_net.parameters(), lr=0.015, momentum=0.9, nesterov=True)
            return NeuralNet(mnist_net, optimizer, CrossEntropyLoss(), device=device)

        def cifar_resnet_creator():
            cifar_resnet = CifarResNet18(in_size=(3, 32, 32), out_size=100)
            optimizer = SGD(cifar_resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, eta_min=0, T_max=20)  # 20 for SVHN

            return NeuralNet(cifar_resnet, optimizer, CrossEntropyLoss(), scheduler, device=device)

        def resnet18_creator():
            resnet = torchvision.models.resnet18(pretrained=True)
            fc1 = resnet.fc
            resnet.fc = nn.Sequential(fc1, nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 200))
            #optimizer = Adam(resnet.parameters())
            optimizer = SGD(resnet.parameters(), lr=0.001, momentum=0.9)
            #scheduler = CosineAnnealingLR(optimizer, eta_min=0, T_max=100)  # 50 for IMAGENET10
            scheduler = StepLR(optimizer, step_size=17, gamma=0.1)

            return NeuralNet(resnet, optimizer, CrossEntropyLoss(), scheduler, device=device)

        def simple_net_creator(shape, lr, device):
            net = NeuralNet.make_simple_mlp_classifier(shape)
            optimizer = Adam(net.parameters(), lr=lr)

            return NeuralNet(net, optimizer, CrossEntropyLoss(), None, device)

        self.add_algorithm_creator('MNIST-NET', mnist_net_creator)
        self.add_algorithm_creator('CIFAR-RESNET-100', cifar_resnet_creator)
        self.add_algorithm_creator('RES-NET18', resnet18_creator)
        self.add_algorithm_creator('FIXED-128x100', lambda: simple_net_creator(
            shape=(128, 128, 64, 100),
            lr=0.0001,
            device=device
        ))
        self.add_algorithm_creator('FIXED-256x100', lambda: simple_net_creator(
            shape=(256, 256, 128, 64, 100),
            lr=0.0001,
            device=device
        ))
        self.add_algorithm_creator('FIXED-512x100', lambda: simple_net_creator(
            shape=(512, 256, 128, 64, 100),
            lr=0.0001,
            device=device
        ))
        self.add_algorithm_creator('FIXED-256x200', lambda: simple_net_creator(
            shape=(256, 256, 128, 64, 200),
            lr=0.0001,
            device=device
        ))

        self.add_evaluator_creator('OffEval-32x50', lambda: OfflineClassStreamEvaluator(batch_size=32, num_epochs=50, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/mnist2.pth'))
        self.add_evaluator_creator('OffEval-64x50', lambda: OfflineClassStreamEvaluator(batch_size=64, num_epochs=50, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/imgnet10-2.pth'))
        self.add_evaluator_creator('OffEval-128x20', lambda: OfflineClassStreamEvaluator(batch_size=128, num_epochs=20, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/svhn.pth'))
        self.add_evaluator_creator('OffEval-128x100', lambda: OfflineClassStreamEvaluator(batch_size=128, num_epochs=100, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/cifar10-2.pth'))

        self.add_evaluator_creator('OffEval-64x100a', lambda: OfflineClassStreamEvaluator(batch_size=64, num_epochs=100, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/imgnet20a-2f.pth'))
        self.add_evaluator_creator('OffEval-64x100b', lambda: OfflineClassStreamEvaluator(batch_size=64, num_epochs=100, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/imgnet20b-2f.pth'))
        self.add_evaluator_creator('OffEval-64x20', lambda: OfflineClassStreamEvaluator(batch_size=64, num_epochs=20, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/imgnet200b-2f-20-2.pth'))
        self.add_evaluator_creator('OffEval-128x100c', lambda: OfflineClassStreamEvaluator(batch_size=128, num_epochs=100, num_workers=8, logdir_root=logdir_root, model_path='pytorch_models/cifar20c-2f.pth'))

        self.add_evaluator_creator('OffEval-64x100', lambda: OfflineClassStreamEvaluator(batch_size=64, num_epochs=100, num_workers=8, logdir_root=logdir_root))
        self.add_evaluator_creator('OffEval-64x40', lambda: OfflineClassStreamEvaluator(batch_size=64, num_epochs=40, num_workers=8, logdir_root=logdir_root))


def run():
    # ExperimentOfflineExample().run(algorithms=['MNIST-NET'], streams=['MNIST-CI'], evaluators=['OffEval-32x50'])
    # ExperimentOfflineExample().run(algorithms=['MNIST-NET'], streams=['FASHION-CI'], evaluators=['OffEval-32x50'])
    # ExperimentOfflineExample().run(algorithms=['CIFAR-RES-NET'], streams=['SVHN-CI'], evaluators=['OffEval-128x20'])
    # ExperimentOfflineExample().run(algorithms=['CIFAR-RES-NET'], streams=['CIFAR10-CI'], evaluators=['OffEval-128x100'])
    # ExperimentOfflineExample().run(algorithms=['RES-NET18'], streams=['IMAGENET10-CI'], evaluators=['OffEval-64x50'])

    ExperimentOfflineExample().run(algorithms=['FIXED-128x100'], streams=['CIFAR100-PRE10-CI'], evaluators=['OffEval-64x100'])
    # ExperimentOfflineExample().run(algorithms=['FIXED-512x100'], streams=['CIFAR100-PRE100-CI'], evaluators=['OffEval-64x100'])
    ExperimentOfflineExample().run(algorithms=['FIXED-256x200'], streams=['IMAGENET200-PRE20B-CI'], evaluators=['OffEval-64x100'])
    ExperimentOfflineExample().run(algorithms=['FIXED-256x200'], streams=['IMAGENET200-PRE200-CI'], evaluators=['OffEval-64x100'])

    # ExperimentOfflineExample().run(algorithms=['RES-NET18'], streams=['IMAGENET20A-CI'], evaluators=['OffEval-64x100b'])
    # ExperimentOfflineExample().run(algorithms=['RES-NET18'], streams=['IMAGENET20B-CI'], evaluators=['OffEval-64x100b'])
    #
    # ExperimentOfflineExample().run(algorithms=['CIFAR-RES-NET'], streams=['CIFAR20C-CI'], evaluators=['OffEval-128x100c'])


if __name__ == '__main__':
    run()
