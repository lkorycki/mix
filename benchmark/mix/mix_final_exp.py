import os

import matplotlib
from learners.mix import MIX

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import torchvision
import torch.nn as nn

from data.stream import ClassStream
from eval.eval import ClassStreamEvaluator
from eval.experiment import Experiment
from learners.nnet import MnistNet, CifarResNet18
import data.data_collection as data_col
import random
import numpy as np


class MIXFinalExperiment(Experiment):

    def __init__(self, run_label, prefix, device, results_dir, extractor_creator, features_num, replay_buffer_size,
                 epochs, batch_size, disable_inter_contrast, super_batch_classes, n_eval, max_classes):
        super().__init__()
        self.run_label = run_label
        self.prefix = prefix
        self.device = device
        self.results_dir = results_dir
        self.algorithms_labels = []

        self.extractor_creator = extractor_creator
        self.features_num = features_num
        self.replay_buffer_size = replay_buffer_size
        self.epochs = epochs
        self.disable_inter_contrast = disable_inter_contrast
        self.batch_size = batch_size
        self.super_batch_classes = super_batch_classes
        self.n_eval = n_eval
        self.max_classes = max_classes

    def prepare(self):
        # export TF_CPP_MIN_LOG_LEVEL=3
        # ulimit -n 64000

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        device = torch.device(self.device)
        torch.cuda.empty_cache()
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)
        matplotlib.use('Agg')
        logdir_root = f'runs/{self.run_label}'
        print('Device: {0}\nLogdir: {1}'.format(device, logdir_root))

        def add_mix_version(v, params):
            label = get_label(v, self.epochs)
            self.add_algorithm_creator(label, lambda: MIX(
                **params,
                extractor=self.extractor_creator(self.features_num),
                init_method='k_means',
                replay_buffer_size=self.replay_buffer_size,
                comp_select=True,
                use_annealing=False,
                sharp_annealing=False,
                full_cov=False,
                cov_min=0.001,
                epochs=self.epochs,
                batch_size=self.batch_size,
                disable_inter_contrast=self.disable_inter_contrast,
                super_batch_classes=self.super_batch_classes,
                replay_buffer_device=self.device,
                device=self.device
            ))

        for v, params in versions_map.items():
            add_mix_version(v, params)

        self.add_data_creator('MNIST-CI-1.0',
                              lambda: ClassStream(data_col.get('MNIST-TRAIN'), data_col.get('MNIST-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('FASHION-CI-1.0',
                              lambda: ClassStream(data_col.get('FASHION-TRAIN'), data_col.get('FASHION-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('SVHN-CI-1.0',
                              lambda: ClassStream(data_col.get('SVHN-TRAIN'), data_col.get('SVHN-TEST'),
                                                  class_frac=1.0,
                                                  class_size=1))
        self.add_data_creator('CIFAR10-CI-1.0',
                              lambda: ClassStream(data_col.get('CIFAR10-TRAIN'), data_col.get('CIFAR10-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('IMAGENET10-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET10-TRAIN'), data_col.get('IMAGENET10-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('CIFAR20-CI-1.0',
                              lambda: ClassStream(data_col.get('CIFAR20C-TRAIN'), data_col.get('CIFAR20C-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('IMAGENET20A-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET20A-TRAIN'), data_col.get('IMAGENET20A-TEST'),
                                                  class_size=1))
        self.add_data_creator('IMAGENET20B-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET20B-TRAIN'), data_col.get('IMAGENET20B-TEST'),
                                                  class_size=1))
        self.add_data_creator('CIFAR100-PRE10-CI-1.0',
                              lambda: ClassStream(data_col.get('CIFAR100-PRE10-TRAIN'),
                                                  data_col.get('CIFAR100-PRE10-TEST'),
                                                  class_size=1))
        self.add_data_creator('CIFAR100-PRE100-CI-1.0',
                              lambda: ClassStream(data_col.get('CIFAR100-PRE100-TRAIN'),
                                                  data_col.get('CIFAR100-PRE100-TEST'),
                                                  class_size=1))
        self.add_data_creator('IMAGENET200-PRE20B-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET200-PRE20B-TRAIN'),
                                                  data_col.get('IMAGENET200-PRE20B-TEST'),
                                                  class_size=1))
        self.add_data_creator('IMAGENET200-PRE200-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET200-PRE200-TRAIN'),
                                                  data_col.get('IMAGENET200-PRE200-TEST'),
                                                  class_size=1))

        self.add_evaluator_creator('FullIncEval', lambda: ClassStreamEvaluator(shuffle=True, num_epochs=self.epochs,
                                                                               n_eval=self.n_eval, epoch_eval=False,
                                                                               max_classes=self.max_classes,
                                                                               num_workers=0, full_load=True,
                                                                               results_dir=self.results_dir,
                                                                               logdir_root=logdir_root,
                                                                               numpy=False, vis=False, emb_vis=False,
                                                                               emb_epoch_vis=False))


def mnist_net_extractor_creator(features_num):
    mnist_net = MnistNet(in_size=(1, 28, 28))
    mnist_net.fc1 = nn.Identity()
    mnist_net.fc2 = nn.Linear(1600, features_num)
    return mnist_net


def cifar_resnet_extractor_creator(features_num):
    cifar_resnet = CifarResNet18(in_size=(3, 32, 32))
    cifar_resnet.fc1 = nn.Identity()
    cifar_resnet.fc2 = nn.Linear(512, features_num)
    return cifar_resnet


def resnet18_extractor_creator(features_num):
    resnet = torchvision.models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(512, features_num)
    return resnet


def fixed_extractor_creator(features_num):
    return nn.Identity()


versions_map = {
    'v1': {
        'k': 1,
        'loss_type': 'mpr',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'max_component',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v2': {
        'k': 3,
        'loss_type': 'mpr',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'max_component',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v3': {
        'k': 5,
        'loss_type': 'mpr',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'max_component',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v4': {
        'k': 10,
        'loss_type': 'mpr',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'max_component',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v5': {
        'k': 1,
        'loss_type': 'ce',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'softmax',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v6': {
        'k': 3,
        'loss_type': 'ce',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'softmax',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v7': {
        'k': 5,
        'loss_type': 'ce',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'softmax',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    },
    'v8': {
        'k': 10,
        'loss_type': 'ce',
        'inter_tightness': 1e-03,
        'intra_tightness': 1e-02,
        'classification_method': 'softmax',
        'extractor_lr': 1e-04,
        'gmm_lr': 1e-05,
    }
}


def get_label(v, ep):
    label = v + '-k={k}-lt={loss_type}-t={inter_tightness}x{intra_tightness}-clasm={classification_method}' \
                '-lr={extractor_lr}x{gmm_lr}'.format(**versions_map[v]) + f'-ep{ep}'
    return label


all_data = ['mnifas', 'svhn', 'cifar10', 'imagenet10', 'cifar20', 'imagenet20a', 'imagenet20b', 'cifar100p10',
            'cifar100p100', 'imagenet200p20', 'imagenet200p200']
all_versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']


def run(versions, data, run_label, device):
    if versions[0] == 'all':
        versions = all_versions
    if data[0] == 'all':
        data = all_data

    for d in data:
        for v in versions:
            run_label_d = f'{run_label}/{v}'
            prefix = 'final'
            results_dir = f'results/{run_label}/{v}'
            epochs = 25
            batch_size = 64
            replay_buffer_size = 256
            n_eval = 1
            extractor = None
            features_num = 128
            disable_inter_contrast = False
            max_classes = -1

            if d == 'mnifas':
                extractor = mnist_net_extractor_creator
            if d in ['svhn', 'cifar10', 'cifar20']:
                extractor = cifar_resnet_extractor_creator
            elif d in ['imagenet10', 'imagenet20a', 'imagenet20b']:
                batch_size = 32
                replay_buffer_size = 128
                extractor = resnet18_extractor_creator
            elif d in ['cifar100p10', 'cifar100p100']:
                replay_buffer_size = 0
                extractor = fixed_extractor_creator
                disable_inter_contrast = True
                features_num = 128 if d == 'cifar100p10' else 512
                n_eval = 10
            elif d in ['imagenet200p20', 'imagenet200p200']:
                batch_size = 32
                replay_buffer_size = 0
                extractor = fixed_extractor_creator
                disable_inter_contrast = True
                features_num = 256
                n_eval = 10
            elif d != 'mnifas':
                raise ValueError(f'Data: {d} unknown/unsupported!')

            streams = []
            if d == 'mnifas':
                streams, epochs = ['MNIST-CI-1.0', 'FASHION-CI-1.0'], 10
            elif d == 'svhn':
                streams = ['SVHN-CI-1.0']
            elif d == 'cifar10':
                streams = ['CIFAR10-CI-1.0']
            elif d == 'imagenet10':
                streams = ['IMAGENET10-CI-1.0']
            elif d == 'cifar20':
                streams = ['CIFAR20-CI-1.0']
            elif d == 'imagenet20a':
                streams = ['IMAGENET20A-CI-1.0']
            elif d == 'imagenet20b':
                streams = ['IMAGENET20B-CI-1.0']
            elif d == 'cifar100p10':
                streams, epochs = ['CIFAR100-PRE10-CI-1.0'], 20
            elif d == 'cifar100p100':
                streams, epochs = ['CIFAR100-PRE100-CI-1.0'], 20
            elif d == 'imagenet200p20':
                streams, epochs = ['IMAGENET200-PRE20B-CI-1.0'], 20
            elif d == 'imagenet200p200':
                streams, epochs = ['IMAGENET200-PRE200-CI-1.0'], 20

            mixe = MIXFinalExperiment(
                run_label=run_label_d,
                prefix=prefix,
                device=device,
                results_dir=results_dir,
                extractor_creator=extractor,
                features_num=features_num,
                replay_buffer_size=replay_buffer_size,
                epochs=epochs,
                batch_size=batch_size,
                disable_inter_contrast=disable_inter_contrast,
                super_batch_classes=10,
                n_eval=n_eval,
                max_classes=max_classes
            )

            mixe.run(algorithms=[get_label(v, epochs)], streams=streams, evaluators=['FullIncEval'])
