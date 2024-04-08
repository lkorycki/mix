import itertools
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


class MIXExperiment(Experiment):

    def __init__(self, run_label, prefix, device, results_dir, k_list, extractor_creator_list, features_num_list,
                 init_method_list, replay_buffer_size_list, comp_select_list, loss_type_list, inter_tightness_list,
                 intra_tightness_list, full_cov_list, cov_min_list, classification_method_list,
                 extractor_lr_list, gmm_lr_list, epochs, batch_size_list, super_batch_classes_list):
        super().__init__()
        self.run_label = run_label
        self.prefix = prefix
        self.device = device
        self.results_dir = results_dir
        self.algorithms_labels = []

        self.k_list = k_list
        self.extractor_creator_list = extractor_creator_list
        self.features_num_list = features_num_list
        self.init_method_list = init_method_list
        self.replay_buffer_size_list = replay_buffer_size_list
        self.comp_select_list = comp_select_list
        self.loss_type_list = loss_type_list
        self.inter_tightness_list = inter_tightness_list
        self.intra_tightness_list = intra_tightness_list
        self.full_cov_list = full_cov_list
        self.cov_min_list = cov_min_list
        self.classification_method_list = classification_method_list
        self.extractor_lr_list = extractor_lr_list
        self.gmm_lr_list = gmm_lr_list
        self.epochs_list = [epochs]  # one value only for the evaluator
        self.batch_size_list = batch_size_list
        self.super_batch_classes_list = super_batch_classes_list

    def prepare(self):
        # export TF_CPP_MIN_LOG_LEVEL=3
        # ulimit -n 64000

        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)

        device = torch.device(self.device)
        torch.cuda.empty_cache()
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)
        matplotlib.use('Agg')
        logdir_root = f'runs/{self.run_label}'
        print('Device: {0}\nLogdir: {1}'.format(device, logdir_root))

        def algorithm_creator(k, extractor_creator, features_num, init_method, replay_buffer_size, comp_select, loss_type,
                              inter_tightness, intra_tightness, full_cov, cov_min, classification_method,
                              extractor_lr, gmm_lr, epochs, batch_size, super_batch_classes):
            return lambda: MIX(
                k=k,
                extractor=extractor_creator(features_num),
                init_method=init_method,
                replay_buffer_size=replay_buffer_size,
                comp_select=comp_select,
                loss_type=loss_type,
                inter_tightness=inter_tightness,
                intra_tightness=intra_tightness,
                use_annealing=False,
                sharp_annealing=False,
                full_cov=full_cov,
                cov_min=cov_min,
                classification_method=classification_method,
                extractor_lr=extractor_lr,
                gmm_lr=gmm_lr,
                epochs=epochs,
                batch_size=batch_size,
                super_batch_classes=super_batch_classes,
                replay_buffer_device=device,
                device=device
            )

        for k, ec, fn, im, rbs, cs, lt, inter, intra, fc, cm, clasm, elr, glr, ep, bs, sbc in itertools.product(
                self.k_list, self.extractor_creator_list, self.features_num_list, self.init_method_list,
                self.replay_buffer_size_list, self.comp_select_list, self.loss_type_list, self.inter_tightness_list,
                self.intra_tightness_list, self.full_cov_list, self.cov_min_list, self.classification_method_list,
                self.extractor_lr_list, self.gmm_lr_list, self.epochs_list, self.batch_size_list,
                self.super_batch_classes_list):

            alg_label = f'{self.prefix}#MIX-k={k}-f={fn}-im={im}-rbs={rbs}-cs={cs}-lt={lt}-t={inter}x{intra}' \
                        f'-fc={fc}-cm={cm}-clasm={clasm}-lr={elr}x{glr}-ep={ep}-bs={bs}-sbc={sbc}'
            self.algorithms_labels.append(alg_label)
            print('Adding: ', alg_label)

            self.add_algorithm_creator(
                alg_label,
                algorithm_creator(k, ec, fn, im, rbs, cs, lt, inter, intra, fc, cm, clasm, elr, glr, ep, bs, sbc)
            )

        self.add_data_creator('MNIST-CI-1.0',
                              lambda: ClassStream(data_col.get('MNIST-TRAIN'), data_col.get('MNIST-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('FASHION-CI-1.0',
                              lambda: ClassStream(data_col.get('FASHION-TRAIN'), data_col.get('FASHION-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))
        self.add_data_creator('SVHN-CI-0.5',
                              lambda: ClassStream(data_col.get('SVHN-TRAIN'), data_col.get('SVHN-TEST'),
                                                  class_frac=0.5,
                                                  class_size=1))
        self.add_data_creator('CIFAR10-CI-0.5',
                              lambda: ClassStream(data_col.get('CIFAR10-TRAIN'), data_col.get('CIFAR10-TEST'),
                                                  class_size=1,
                                                  class_frac=0.5))
        self.add_data_creator('IMAGENET10-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET10-TRAIN'), data_col.get('IMAGENET10-TEST'),
                                                  class_size=1,
                                                  class_frac=1.0))

        self.add_evaluator_creator('FullIncEval', lambda: ClassStreamEvaluator(shuffle=True, num_epochs=self.epochs_list[0],
                                                                               num_workers=0, full_load=True, results_dir=self.results_dir,
                                                                               logdir_root=logdir_root, max_classes=-1,
                                                                               numpy=False, vis=False, emb_vis=False, emb_epoch_vis=False))


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


all_data = ['mnifas', 'svhn', 'cifar10', 'imagenet10']
all_params = ['learning_rates', 'tightness', 'loss_updates', 'replay_buffers', 'covs', 'ks']


def run(params, data, run_label, device):
    if data[0] == 'all':
        data = all_data
    if params[0] == 'all':
        params = all_params

    for param in params:
        for d in data:
            mixe: MIXExperiment = MIXExperiment(
                run_label='',
                prefix='',
                device=device,
                results_dir='',
                k_list=[3],
                extractor_creator_list=[mnist_net_extractor_creator],
                features_num_list=[128],
                init_method_list=['k_means'],
                replay_buffer_size_list=[256],
                comp_select_list=[True],
                loss_type_list=['mpr'],
                inter_tightness_list=[0.002],
                intra_tightness_list=[0.01],
                full_cov_list=[False],
                cov_min_list=[0.001],
                classification_method_list=['max_component'],
                extractor_lr_list=[1e-04],
                gmm_lr_list=[1e-03],
                epochs=10,
                batch_size_list=[64],
                super_batch_classes_list=[10]
            )

            mixe.run_label = f'{run_label}/{param}/{d}'
            mixe.prefix = param
            mixe.results_dir = f'results/{run_label}/{param}/{d}'

            if d in ['svhn', 'cifar10']:
                mixe.inter_tightness_list = [0.0002]
                mixe.intra_tightness_list = [0.001]
                mixe.epochs_list = [25]
                mixe.extractor_creator_list = [cifar_resnet_extractor_creator]
            elif d == 'imagenet10':
                mixe.inter_tightness_list = [0.0002]
                mixe.intra_tightness_list = [0.001]
                mixe.epochs_list = [25]
                mixe.batch_size_list = [32]
                mixe.replay_buffer_size_list = [128]
                mixe.extractor_creator_list = [resnet18_extractor_creator]
            elif d != 'mnifas':
                raise ValueError(f'Data: {d} unknown/unsupported!')

            streams = []
            if d == 'mnifas':
                streams = ['MNIST-CI-1.0', 'FASHION-CI-1.0']
            elif d == 'svhn':
                streams = ['SVHN-CI-0.5']
            elif d == 'cifar10':
                streams = ['CIFAR10-CI-0.5']
            elif d == 'imagenet10':
                streams = ['IMAGENET10-CI-1.0']

            if param == 'learning_rates':
                mixe.extractor_lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                mixe.gmm_lr_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            elif param == 'tightness':
                mixe.inter_tightness_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
                mixe.intra_tightness_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            elif param == 'loss_updates':
                mixe.loss_type_list = ['ce', 'mp', 'mpr']
                mixe.classification_method_list = ['max_component', 'softmax']
            elif param == 'replay_buffers':
                mixe.replay_buffer_size_list = [8, 64, 128, 256, 512]
            elif param == 'covs':
                mixe.full_cov_list = [True, False]
            elif param == 'ks':
                mixe.k_list = [1, 3, 5, 10, 20]
            else:
                raise ValueError(f'Param: {param} unknown!')

            mixe.run(algorithms=mixe.algorithms_labels, streams=streams, evaluators=['FullIncEval'])

