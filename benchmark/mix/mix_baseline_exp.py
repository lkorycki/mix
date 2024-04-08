import os

import matplotlib
import torch
import random
import numpy as np
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import data.data_collection as data_col
from data.stream import ClassStream
from eval.eval import ClassStreamEvaluator, AvalancheStreamEvaluator, OfflineClassStreamEvaluator
from eval.experiment import Experiment
from learners.er import ClassIncrementalExperienceReplay, StreamingExperienceReplay, SubspaceBuffer
from learners.nnet import MnistNet, CifarResNet18, NeuralNet, ConvNeuralNet
from learners.off import OfflinePredictor
from learners.wrappers.ava import ICaRL, GSSGreedy, SI, LWF
from learners.wrappers.mth import AGEM, DER

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class MIXBaselineExperiment(Experiment):

    def __init__(self, run_label, prefix, device, results_dir, lr, bn, epochs, batch_size, n_eval, r):
        super().__init__()
        self.run_label = run_label
        self.prefix = prefix
        self.device = device
        self.results_dir = results_dir
        self.algorithms_labels = []

        self.lr = lr
        self.bn = bn
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_eval = n_eval

        self.r = r

    def prepare(self):
        # export TF_CPP_MIN_LOG_LEVEL=3
        # ulimit -n 64000

        torch.manual_seed(self.r)
        random.seed(self.r)
        np.random.seed(self.r)

        device = torch.device(self.device)
        torch.cuda.empty_cache()
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)
        matplotlib.use('Agg')
        logdir_root = f'runs/{self.run_label}'
        print('Device: {0}\nLogdir: {1}'.format(device, logdir_root))

        self.add_er_algorithms(self.lr, self.bn)
        self.add_naive_algorithms(self.lr, self.bn)
        self.add_ersb_algorithms(self.lr, self.bn)
        self.add_icarl_algorithms(self.lr, self.bn)
        self.add_gss_algorithms(self.lr, self.bn)
        self.add_si_algorithms(self.lr, self.bn)
        self.add_lwf_algorithms(self.lr, self.bn)
        self.add_agem_algorithms(self.lr, self.bn)
        self.add_der_algorithms(self.lr, self.bn)
        self.add_offline_algorithms(self.lr, self.epochs)

        self.add_data_creator('MNIST-CI-1.0',
                              lambda: ClassStream(data_col.get('MNIST-TRAIN'), data_col.get('MNIST-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('FASHION-CI-1.0',
                              lambda: ClassStream(data_col.get('FASHION-TRAIN'), data_col.get('FASHION-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('SVHN-CI-1.0',
                              lambda: ClassStream(data_col.get('SVHN-TRAIN'), data_col.get('SVHN-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('CIFAR10-CI-1.0',
                              lambda: ClassStream(data_col.get('CIFAR10-TRAIN'), data_col.get('CIFAR10-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('IMAGENET10-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET10-TRAIN'), data_col.get('IMAGENET10-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('CIFAR20-CI-1.0',
                              lambda: ClassStream(data_col.get('CIFAR20C-TRAIN'), data_col.get('CIFAR20C-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('IMAGENET20A-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET20A-TRAIN'), data_col.get('IMAGENET20A-TEST'),
                                                  class_size=1, class_frac=1.0))
        self.add_data_creator('IMAGENET20B-CI-1.0',
                              lambda: ClassStream(data_col.get('IMAGENET20B-TRAIN'), data_col.get('IMAGENET20B-TEST'),
                                                  class_size=1, class_frac=1.0))
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

        self.add_evaluator_creator(f'FullIncEval-ep{self.epochs}',
                                   lambda: ClassStreamEvaluator(shuffle=True, num_epochs=self.epochs,
                                                                n_eval=self.n_eval,
                                                                num_workers=0,
                                                                epoch_eval=False,
                                                                full_load=True,
                                                                results_dir=self.results_dir,
                                                                logdir_root=logdir_root,
                                                                max_classes=-1,
                                                                numpy=False, vis=False,
                                                                emb_vis=False,
                                                                emb_epoch_vis=False))

        self.add_evaluator_creator(f'IncEval-ep{self.epochs}-bs{self.batch_size}',
                                   lambda: ClassStreamEvaluator(batch_size=self.batch_size, n_eval=self.n_eval,
                                                                shuffle=True,
                                                                epoch_eval=False,
                                                                num_epochs=self.epochs, num_workers=0,
                                                                logdir_root=logdir_root,
                                                                results_dir=self.results_dir,
                                                                vis=False))

        self.add_evaluator_creator(f'AvaEval', lambda: AvalancheStreamEvaluator(num_workers=0, n_eval=self.n_eval,
                                                                                results_dir=self.results_dir,
                                                                                logdir_root=logdir_root,
                                                                                max_classes=-1,
                                                                                numpy=False, vis=False,
                                                                                emb_vis=False))

        self.add_evaluator_creator(f'OfflineEval-ep{self.epochs}-bs{self.batch_size}',
                                   lambda: OfflineClassStreamEvaluator(batch_size=self.batch_size, n_eval=self.n_eval,
                                                                       num_epochs=self.epochs,
                                                                       num_workers=0,
                                                                       logdir_root=logdir_root,
                                                                       numpy=False, vis=False))

    def add_er_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-ER-b256-adam-bs64-c10-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.mnist_net_creator(lr=lr, num_classes=10, bn=bn),
                                       replay_buffer_size=256,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'CIFRESNET-ER-b256-adam-bs64-c10-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.cifar_resnet_creator(lr=lr, num_classes=10, bn=bn),
                                       replay_buffer_size=256,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'CIFRESNET-ER-b256-adam-bs64-c20-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.cifar_resnet_creator(lr=lr, num_classes=20, bn=bn),
                                       replay_buffer_size=256,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'RESNET18-ER-b128-adam-bs32-c10-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.resnet18_creator(lr=lr, num_classes=10, bn=bn),
                                       replay_buffer_size=128,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'RESNET18-ER-b128-adam-bs32-c20-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.resnet18_creator(lr=lr, num_classes=20, bn=bn),
                                       replay_buffer_size=128,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-ER-b1-adam-bs64-c100-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((128, 100)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=1,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-ER-b1-adam-bs64-c100-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((512, 100)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=1,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-ER-b1-adam-bs32-c200-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((256, 200)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=1,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-ER-b1-adam-bs32-c200-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((256, 200)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=1,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

    def add_naive_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-NAIVE-adam-bs64-c10-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.mnist_net_creator(lr=lr, num_classes=10, bn=bn),
                                       replay_buffer_size=0,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'CIFRESNET-NAIVE-adam-bs64-c10-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.cifar_resnet_creator(lr=lr, num_classes=10, bn=bn),
                                       replay_buffer_size=0,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'CIFRESNET-NAIVE-adam-bs64-c20-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.cifar_resnet_creator(lr=lr, num_classes=20, bn=bn),
                                       replay_buffer_size=0,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'RESNET18-NAIVE-adam-bs32-c10-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.resnet18_creator(lr=lr, num_classes=10, bn=bn),
                                       replay_buffer_size=0,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'RESNET18-NAIVE-adam-bs32-c20-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.resnet18_creator(lr=lr, num_classes=20, bn=bn),
                                       replay_buffer_size=0,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-NAIVE-adam-bs64-c100-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((128, 100)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=0,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-NAIVE-adam-bs64-c100-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((512, 100)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=0,
                                       batch_size=64,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-NAIVE-adam-bs32-c200-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((256, 200)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=0,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-NAIVE-adam-bs32-c200-lr{lr}-bn{bn}',
                                   lambda: ClassIncrementalExperienceReplay(
                                       **self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((256, 200)),
                                           lr=lr
                                       ),
                                       replay_buffer_size=0,
                                       batch_size=32,
                                       replay_buffer_device=self.device,
                                       device=self.device
                                   ))

    def add_ersb_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-ERSB-b10x25-adam-c10-lr{lr}-bn{bn}', lambda: StreamingExperienceReplay(
            NeuralNet(**self.mnist_net_creator(lr=lr, num_classes=10, bn=bn), loss=CrossEntropyLoss(),
                      device=self.device),
            SubspaceBuffer(max_centroids=10, max_instances=25, mode='class_sample'),
            mbs=64
        ))

        self.add_algorithm_creator(f'CIFRESNET-ERSB-b10x25-adam-c10-lr{lr}-bn{bn}', lambda: StreamingExperienceReplay(
            NeuralNet(**self.cifar_resnet_creator(lr=lr, num_classes=10, bn=bn), loss=CrossEntropyLoss(),
                      device=self.device),
            SubspaceBuffer(max_centroids=10, max_instances=25, mode='class_sample'),
            mbs=64,
        ))

        self.add_algorithm_creator(f'CIFRESNET-ERSB-b10x25-adam-c20-lr{lr}-bn{bn}', lambda: StreamingExperienceReplay(
            NeuralNet(**self.cifar_resnet_creator(lr=lr, num_classes=20, bn=bn), loss=CrossEntropyLoss(),
                      device=self.device),
            SubspaceBuffer(max_centroids=10, max_instances=25, mode='class_sample'),
            mbs=64,
        ))

        self.add_algorithm_creator(f'RESNET18-ERSB-b10x15-adam-c10-lr{lr}-bn{bn}', lambda: StreamingExperienceReplay(
            NeuralNet(**self.resnet18_creator(lr=lr, num_classes=10, bn=bn), loss=CrossEntropyLoss(),
                      device=self.device),
            SubspaceBuffer(max_centroids=10, max_instances=15, mode='class_sample'),
            mbs=32
        ))

        self.add_algorithm_creator(f'RESNET18-ERSB-b10x15-adam-c20-lr{lr}-bn{bn}', lambda: StreamingExperienceReplay(
            NeuralNet(**self.resnet18_creator(lr=lr, num_classes=20, bn=bn), loss=CrossEntropyLoss(),
                      device=self.device),
            SubspaceBuffer(max_centroids=10, max_instances=15, mode='class_sample'),
            mbs=32
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-ERSB-b1x1-adam-c100-lr{lr}-bn{bn}',
                                   lambda: StreamingExperienceReplay(
                                       NeuralNet(**self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((128, 100)),
                                           lr=lr
                                       ), loss=CrossEntropyLoss(), device=self.device),
                                       SubspaceBuffer(max_centroids=1, max_instances=1, mode='class_sample'),
                                       mbs=64
                                   ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-ERSB-b1x1-adam-c100-lr{lr}-bn{bn}',
                                   lambda: StreamingExperienceReplay(
                                       NeuralNet(**self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((512, 100)),
                                           lr=lr
                                       ), loss=CrossEntropyLoss(), device=self.device),
                                       SubspaceBuffer(max_centroids=1, max_instances=1, mode='class_sample'),
                                       mbs=64
                                   ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-ERSB-b1x1-adam-c200-lr{lr}-bn{bn}',
                                   lambda: StreamingExperienceReplay(
                                       NeuralNet(**self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((256, 200)),
                                           lr=lr
                                       ), loss=CrossEntropyLoss(), device=self.device),
                                       SubspaceBuffer(max_centroids=1, max_instances=1, mode='class_sample'),
                                       mbs=32
                                   ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-ERSB-b1x1-adam-c200-lr{lr}-bn{bn}',
                                   lambda: StreamingExperienceReplay(
                                       NeuralNet(**self.conv_net_creator(
                                           nn.Identity(),
                                           NeuralNet.make_simple_mlp_classifier((256, 200)),
                                           lr=lr
                                       ), loss=CrossEntropyLoss(), device=self.device),
                                       SubspaceBuffer(max_centroids=1, max_instances=1, mode='class_sample'),
                                       mbs=32
                                   ))

    def add_icarl_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-ICARL-b256-adam-c10-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                self.mnist_net_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            epochs=10,
            memory_size=256,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-ICARL-b256-adam-c10-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            epochs=10,
            memory_size=256,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-ICARL-b256-adam-c20-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            epochs=20,
            memory_size=256,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-ICARL-b128-adam-c10-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            epochs=10,
            memory_size=128,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-ICARL-b128-adam-c20-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            epochs=20,
            memory_size=128,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-ICARL-b1-adam-c100-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((128, 100)),
                lr=lr
            ),
            epochs=20,
            memory_size=1,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-ICARL-b1-adam-c100-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((512, 100)),
                lr=lr
            ),
            epochs=20,
            memory_size=1,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-ICARL-b1-adam-c200-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            epochs=20,
            memory_size=1,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-ICARL-b1-adam-c200-lr{lr}-bn{bn}', lambda: ICaRL(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            epochs=20,
            memory_size=1,
            batch_size=32,
            device=self.device
        ))

    def add_gss_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-GSS-b10x256-adam-c10-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                self.mnist_net_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=10,
            input_size=[1, 28, 28],
            memory_size=10 * 256,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-GSS-b10x256-adam-c10-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=10,
            input_size=[3, 32, 32],
            memory_size=10 * 256,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-GSS-b20x256-adam-c20-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=20,
            input_size=[3, 32, 32],
            memory_size=20 * 256,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-GSS-b10x128-adam-c10-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=10,
            memory_size=10 * 128,
            input_size=[3, 224, 224],
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-GSS-b20x128-adam-c20-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=20,
            memory_size=20 * 128,
            input_size=[3, 224, 224],
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-GSS-b100x1-adam-c100-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((128, 100)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=20,
            input_size=[128],
            memory_size=100 * 1,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-GSS-b100x1-adam-c100-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((512, 100)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=20,
            input_size=[512],
            memory_size=100 * 1,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-GSS-b200x1-adam-c200-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=20,
            memory_size=200 * 1,
            input_size=[256],
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-GSS-b200x1-adam-c200-lr{lr}-bn{bn}', lambda: GSSGreedy(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            epochs=20,
            memory_size=200 * 1,
            input_size=[256],
            batch_size=32,
            device=self.device
        ))

    def add_si_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-SI-lb0.0001-adam-c10-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                self.mnist_net_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=10,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-SI-lb0.0001-adam-c10-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=10,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-SI-lb0.0001-adam-c20-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=20,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-SI-lb0.0001-adam-c10-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=10,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-SI-lb0.0001-adam-c20-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=20,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-SI-lb0.0001-adam-c100-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((128, 100)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=20,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-SI-lb0.0001-adam-c100-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((512, 100)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=20,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-SI-lb0.0001-adam-c200-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=20,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-SI-lb0.0001-adam-c200-lr{lr}-bn{bn}', lambda: SI(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            si_lambda=0.0001,
            epochs=20,
            batch_size=32,
            device=self.device
        ))

    def add_lwf_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-LWF-t2-la-adam-c10-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                self.mnist_net_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(10)],
            epochs=10,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-LWF-t2-la-adam-c10-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(10)],
            epochs=10,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-LWF-t2-la-adam-c20-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(20)],
            epochs=20,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-LWF-t2-la-adam-c10-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(10)],
            epochs=10,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-LWF-t2-la-adam-c20-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(20)],
            epochs=20,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-LWF-t2-la-adam-c100-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((128, 100)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(100)],
            epochs=20,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-LWF-t2-la-adam-c100-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((512, 100)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(100)],
            epochs=20,
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-LWF-t2-la-adam-c200-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(200)],
            epochs=20,
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-LWF-t2-la-adam-c200-lr{lr}-bn{bn}', lambda: LWF(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            criterion=CrossEntropyLoss(),
            temperature=2.0,
            alpha=[1.0 - (1.0 / (i + 1.0)) for i in range(200)],
            epochs=20,
            batch_size=32,
            device=self.device
        ))

    def add_agem_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-AGEM-b10x256-mbs64-adam-c10-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                self.mnist_net_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 128, 10)),
                lr=lr
            ),
            buffer_size=10 * 256,
            minibatch_size=64,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-AGEM-b10x256-mbs64-adam-c10-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 128, 10)),
                lr=lr
            ),
            buffer_size=10 * 256,
            minibatch_size=64,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-AGEM-b20x256-mbs64-adam-c20-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 128, 20)),
                lr=lr
            ),
            buffer_size=20 * 256,
            minibatch_size=64,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-AGEM-b10x128-mbs32-adam-c10-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            buffer_size=10 * 128,
            minibatch_size=32,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-AGEM-b20x128-mbs32-adam-c20-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            buffer_size=20 * 128,
            minibatch_size=32,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-AGEM-b100x1-mbs64-adam-c100-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((128, 100)),
                lr=lr
            ),
            buffer_size=100 * 1,
            minibatch_size=64,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-AGEM-b100x1-mbs64-adam-c100-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((512, 100)),
                lr=lr
            ),
            buffer_size=100 * 1,
            minibatch_size=64,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=64,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-AGEM-b200x1-mbs32-adam-c200-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            buffer_size=200 * 1,
            minibatch_size=32,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=32,
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-AGEM-b200x1-mbs32-adam-c200-lr{lr}-bn{bn}', lambda: AGEM(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            buffer_size=200 * 1,
            minibatch_size=32,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            batch_size=32,
            device=self.device
        ))

    def add_der_algorithms(self, lr, bn):
        self.add_algorithm_creator(f'MNISTNET-DER-b10x256-mbs64-a0.5-adam-c10-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                self.mnist_net_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 128, 10)),
                lr=lr
            ),
            minibatch_size=64,
            alpha=0.5,
            buffer_size=10 * 256,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-DER-b10x256-mbs64-a0.5-adam-c10-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 128, 10)),
                lr=lr
            ),
            minibatch_size=64,
            alpha=0.5,
            buffer_size=10 * 256,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'CIFRESNET-DER-b20x256-mbs64-a0.5-adam-c20-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                self.cifar_resnet_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 128, 20)),
                lr=lr
            ),
            minibatch_size=64,
            alpha=0.5,
            buffer_size=20 * 256,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-DER-b10x128-mbs32-a0.5-adam-c10-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 10)),
                lr=lr
            ),
            minibatch_size=32,
            alpha=0.5,
            buffer_size=10 * 128,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'RESNET18-DER-b20x128-mbs32-a0.5-adam-c20-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                self.resnet18_extractor_creator(features_num=128, bn=bn),
                NeuralNet.make_simple_mlp_classifier((128, 20)),
                lr=lr
            ),
            minibatch_size=32,
            alpha=0.5,
            buffer_size=20 * 128,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p10-DER-b100x1-mbs64-adam-c100-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((128, 100)),
                lr=lr
            ),
            minibatch_size=64,
            alpha=0.5,
            buffer_size=100 * 1,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-CIFAR100p100-DER-b100x1-mbs64-adam-c100-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((512, 100)),
                lr=lr
            ),
            minibatch_size=64,
            alpha=0.5,
            buffer_size=100 * 1,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p20-DER-b200x1-mbs32-adam-c200-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            minibatch_size=32,
            alpha=0.5,
            buffer_size=200 * 1,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

        self.add_algorithm_creator(f'FIXED-IMGNET200p200-DER-b200x1-mbs32-adam-c200-lr{lr}-bn{bn}', lambda: DER(
            **self.conv_net_creator(
                nn.Identity(),
                NeuralNet.make_simple_mlp_classifier((256, 200)),
                lr=lr
            ),
            minibatch_size=32,
            alpha=0.5,
            buffer_size=200 * 1,
            loss=CrossEntropyLoss(),
            buffer_mode='reservoir',
            device=self.device
        ))

    def add_offline_algorithms(self, lr, ep):
        self.add_algorithm_creator(f'RESNET18-OFFLINE-IMGNET20A-lr{lr}-ep{ep}', lambda: self.offline_resnet18_creator(
            'pytorch_models/imgnet20a-2f.pth', self.device))
        self.add_algorithm_creator(f'RESNET18-OFFLINE-IMGNET20B-lr{lr}-ep{ep}', lambda: self.offline_resnet18_creator(
            'pytorch_models/imgnet20b-2f.pth', self.device))

        self.add_algorithm_creator(f'FIXED-OFFLINE-CIFAR100p10-lr{lr}-ep{ep}', lambda: self.simple_net_creator(
            shape=(128, 128, 64, 100),
            lr=lr,
            device=self.device
        ))
        self.add_algorithm_creator(f'FIXED-OFFLINE-CIFAR100p100-lr{lr}-ep{ep}', lambda: self.simple_net_creator(
            shape=(512, 256, 128, 64, 100),
            lr=lr,
            device=self.device
        ))
        self.add_algorithm_creator(f'FIXED-OFFLINE-IMGNET200p20-lr{lr}-ep{ep}', lambda: self.simple_net_creator(
            shape=(256, 256, 128, 64, 200),
            lr=lr,
            device=self.device
        ))
        self.add_algorithm_creator(f'FIXED-OFFLINE-IMGNET200p200-lr{lr}-ep{ep}', lambda: self.simple_net_creator(
            shape=(256, 256, 128, 64, 200),
            lr=lr,
            device=self.device
        ))

    @staticmethod
    def offline_resnet18_creator(model_path, device):
        resnet18 = torchvision.models.resnet18(pretrained=True)
        fc1 = resnet18.fc
        resnet18.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
        resnet18.load_state_dict(torch.load(model_path))

        return OfflinePredictor(resnet18, device)

    @staticmethod
    def mnist_net_creator(lr, num_classes, bn, fixed=False):
        mnist_net = MnistNet(in_size=(1, 28, 28), out_size=num_classes, bn=bn)
        if fixed:
            mnist_net.feature_extractor = nn.Identity()
        optimizer = Adam(mnist_net.parameters(), lr=lr)
        scheduler = None

        return {'model': mnist_net, 'optimizer': optimizer, 'scheduler': scheduler}

    @staticmethod
    def cifar_resnet_creator(lr, num_classes, bn, fixed=False):
        cifar_resnet = CifarResNet18(in_size=(3, 32, 32), out_size=num_classes, bn=bn)
        if fixed:
            cifar_resnet.feature_extractor = nn.Identity()
        optimizer = Adam(cifar_resnet.parameters(), lr=lr)
        scheduler = None

        return {'model': cifar_resnet, 'optimizer': optimizer, 'scheduler': scheduler}

    @staticmethod
    def resnet18_creator(lr, num_classes, bn, fixed=False):
        norm_layer = None if bn else nn.Identity
        resnet = torchvision.models.resnet18(pretrained=False, norm_layer=norm_layer)
        if fixed:
            resnet.conv1 = nn.Identity()
            resnet.bn1 = nn.Identity()
            resnet.relu = nn.Identity()
            resnet.maxpool = nn.Identity()
            resnet.layer1 = nn.Identity()
            resnet.layer2 = nn.Identity()
            resnet.layer3 = nn.Identity()
            resnet.layer4 = nn.Identity()
            resnet.avgpool = nn.Identity()
        resnet.fc = nn.Linear(512, num_classes)
        optimizer = Adam(resnet.parameters(), lr=lr)
        scheduler = None

        return {'model': resnet, 'optimizer': optimizer, 'scheduler': scheduler}

    @staticmethod
    def mnist_net_extractor_creator(features_num, bn):
        mnist_net = MnistNet(in_size=(1, 28, 28), bn=bn)
        mnist_net.fc1 = nn.Identity()
        mnist_net.fc2 = nn.Linear(1600, features_num)
        return mnist_net

    @staticmethod
    def cifar_resnet_extractor_creator(features_num, bn):
        cifar_resnet = CifarResNet18(in_size=(3, 32, 32), bn=bn)
        cifar_resnet.fc1 = nn.Identity()
        cifar_resnet.fc2 = nn.Linear(512, features_num)
        return cifar_resnet

    @staticmethod
    def resnet18_extractor_creator(features_num, bn):
        norm_layer = None if bn else nn.Identity
        resnet = torchvision.models.resnet18(pretrained=False, norm_layer=norm_layer)
        resnet.fc = nn.Linear(512, features_num)
        return resnet

    @staticmethod
    def conv_net_creator(extractor, classifier, lr):
        net = ConvNeuralNet(extractor, classifier)
        optimizer = Adam(net.parameters(), lr=lr)
        scheduler = None

        return {'model': net, 'optimizer': optimizer, 'scheduler': scheduler}

    @staticmethod
    def optimizer_wrapper(extractor, lr):
        return {'extractor': extractor, 'optimizer': Adam(extractor.parameters(), lr)}

    @staticmethod
    def simple_net_creator(shape, lr, device):
        net = NeuralNet.make_simple_mlp_classifier(shape)
        optimizer = Adam(net.parameters(), lr=lr)

        return NeuralNet(net, optimizer, CrossEntropyLoss(), None, device)


algs_mapping = {
    'er': {
        'mnifas': 'MNISTNET-ER-b256-adam-bs64-c10',
        'svhn': 'CIFRESNET-ER-b256-adam-bs64-c10',
        'cifar10': 'CIFRESNET-ER-b256-adam-bs64-c10',
        'imagenet10': 'RESNET18-ER-b128-adam-bs32-c10',
        'cifar20': 'CIFRESNET-ER-b256-adam-bs64-c20',
        'imagenet20a': 'RESNET18-ER-b128-adam-bs32-c20',
        'imagenet20b': 'RESNET18-ER-b128-adam-bs32-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-ER-b1-adam-bs64-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-ER-b1-adam-bs64-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-ER-b1-adam-bs32-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-ER-b1-adam-bs32-c200'
    },
    'naive': {
        'mnifas': 'MNISTNET-NAIVE-adam-bs64-c10',
        'svhn': 'CIFRESNET-NAIVE-adam-bs64-c10',
        'cifar10': 'CIFRESNET-NAIVE-adam-bs64-c10',
        'imagenet10': 'RESNET18-NAIVE-adam-bs32-c10',
        'cifar20': 'CIFRESNET-NAIVE-adam-bs64-c20',
        'imagenet20a': 'RESNET18-NAIVE-adam-bs32-c20',
        'imagenet20b': 'RESNET18-NAIVE-adam-bs32-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-NAIVE-adam-bs64-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-NAIVE-adam-bs64-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-NAIVE-adam-bs32-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-NAIVE-adam-bs32-c200'
    },
    'ersb': {
        'mnifas': 'MNISTNET-ERSB-b10x25-adam-c10',
        'svhn': 'CIFRESNET-ERSB-b10x25-adam-c10',
        'cifar10': 'CIFRESNET-ERSB-b10x25-adam-c10',
        'imagenet10': 'RESNET18-ERSB-b10x15-adam-c10',
        'cifar20': 'CIFRESNET-ERSB-b10x25-adam-c20',
        'imagenet20a': 'RESNET18-ERSB-b10x15-adam-c20',
        'imagenet20b': 'RESNET18-ERSB-b10x15-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-ERSB-b1x1-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-ERSB-b1x1-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-ERSB-b1x1-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-ERSB-b1x1-adam-c200'
    },
    'icarl': {
        'mnifas': 'MNISTNET-ICARL-b256-adam-c10',
        'svhn': 'CIFRESNET-ICARL-b256-adam-c10',
        'cifar10': 'CIFRESNET-ICARL-b256-adam-c10',
        'imagenet10': 'RESNET18-ICARL-b128-adam-c10',
        'cifar20': 'CIFRESNET-ICARL-b256-adam-c20',
        'imagenet20a': 'RESNET18-ICARL-b128-adam-c20',
        'imagenet20b': 'RESNET18-ICARL-b128-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-ICARL-b1-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-ICARL-b1-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-ICARL-b1-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-ICARL-b1-adam-c200'
    },
    'gss': {
        'mnifas': 'MNISTNET-GSS-b10x256-adam-c10',
        'svhn': 'CIFRESNET-GSS-b10x256-adam-c10',
        'cifar10': 'CIFRESNET-GSS-b10x256-adam-c10',
        'imagenet10': 'RESNET18-GSS-b10x128-adam-c10',
        'cifar20': 'CIFRESNET-GSS-b20x256-adam-c20',
        'imagenet20a': 'RESNET18-GSS-b20x128-adam-c20',
        'imagenet20b': 'RESNET18-GSS-b20x128-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-GSS-b100x1-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-GSS-b100x1-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-GSS-b200x1-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-GSS-b200x1-adam-c200'
    },
    'si': {
        'mnifas': 'MNISTNET-SI-lb0.0001-adam-c10',
        'svhn': 'CIFRESNET-SI-lb0.0001-adam-c10',
        'cifar10': 'CIFRESNET-SI-lb0.0001-adam-c10',
        'imagenet10': 'RESNET18-SI-lb0.0001-adam-c10',
        'cifar20': 'CIFRESNET-SI-lb0.0001-adam-c20',
        'imagenet20a': 'RESNET18-SI-lb0.0001-adam-c20',
        'imagenet20b': 'RESNET18-SI-lb0.0001-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-SI-lb0.0001-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-SI-lb0.0001-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-SI-lb0.0001-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-SI-lb0.0001-adam-c200'
    },
    'lwf': {
        'mnifas': 'MNISTNET-LWF-t2-la-adam-c10',
        'svhn': 'CIFRESNET-LWF-t2-la-adam-c10',
        'cifar10': 'CIFRESNET-LWF-t2-la-adam-c10',
        'imagenet10': 'RESNET18-LWF-t2-la-adam-c10',
        'cifar20': 'CIFRESNET-LWF-t2-la-adam-c20',
        'imagenet20a': 'RESNET18-LWF-t2-la-adam-c20',
        'imagenet20b': 'RESNET18-LWF-t2-la-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-LWF-t2-la-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-LWF-t2-la-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-LWF-t2-la-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-LWF-t2-la-adam-c200'
    },
    'agem': {
        'mnifas': 'MNISTNET-AGEM-b10x256-mbs64-adam-c10',
        'svhn': 'CIFRESNET-AGEM-b10x256-mbs64-adam-c10',
        'cifar10': 'CIFRESNET-AGEM-b10x256-mbs64-adam-c10',
        'imagenet10': 'RESNET18-AGEM-b10x128-mbs32-adam-c10',
        'cifar20': 'CIFRESNET-AGEM-b20x256-mbs64-adam-c20',
        'imagenet20a': 'RESNET18-AGEM-b20x128-mbs32-adam-c20',
        'imagenet20b': 'RESNET18-AGEM-b20x128-mbs32-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-AGEM-b100x1-mbs64-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-AGEM-b100x1-mbs64-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-AGEM-b200x1-mbs32-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p20-AGEM-b200x1-mbs32-adam-c200'
    },
    'der': {
        'mnifas': 'MNISTNET-DER-b10x256-mbs64-a0.5-adam-c10',
        'svhn': 'CIFRESNET-DER-b10x256-mbs64-a0.5-adam-c10',
        'cifar10': 'CIFRESNET-DER-b10x256-mbs64-a0.5-adam-c10',
        'imagenet10': 'RESNET18-DER-b10x128-mbs32-a0.5-adam-c10',
        'cifar20': 'CIFRESNET-DER-b20x256-mbs64-a0.5-adam-c20',
        'imagenet20a': 'RESNET18-DER-b20x128-mbs32-a0.5-adam-c20',
        'imagenet20b': 'RESNET18-DER-b20x128-mbs32-a0.5-adam-c20',
        'cifar100p10': 'FIXED-CIFAR100p10-DER-b100x1-mbs64-adam-c100',
        'cifar100p100': 'FIXED-CIFAR100p100-DER-b100x1-mbs64-adam-c100',
        'imagenet200p20': 'FIXED-IMGNET200p20-DER-b200x1-mbs32-adam-c200',
        'imagenet200p200': 'FIXED-IMGNET200p200-DER-b200x1-mbs32-adam-c200'
    },
    'offline': {
        'imagenet20a': 'RESNET18-OFFLINE-IMGNET20A',
        'imagenet20b': 'RESNET18-OFFLINE-IMGNET20B',
        'cifar100p10': 'FIXED-OFFLINE-CIFAR100p10',
        'cifar100p100': 'FIXED-OFFLINE-CIFAR100p100',
        'imagenet200p20': 'FIXED-OFFLINE-IMGNET200p20',
        'imagenet200p200': 'FIXED-OFFLINE-IMGNET200p200'
    }
}


def get_alg_mapping(alg, data):
    return algs_mapping[alg][data]


all_baselines = ['naive', 'er', 'ersb', 'icarl', 'gss', 'si', 'lwf', 'agem', 'der']
all_data = ['mnifas', 'svhn', 'cifar10', 'imagenet10', 'cifar20', 'imagenet20a', 'imagenet20b', 'cifar100p10',
            'cifar100p100', 'imagenet200p20', 'imagenet200p200']


def run(algorithms, lrs, bns, data, run_label, device):
    if algorithms[0] == 'all':
        algorithms = all_baselines
    if data[0] == 'all':
        data = all_data

    for alg in algorithms:
        for d in data:
            for bn in bns:
                for lr in lrs:
                    bs, n_eval = 64, 1
                    ep = 25

                    if d == 'mnifas':
                        streams, ep = ['MNIST-CI-1.0', 'FASHION-CI-1.0'], 10
                    elif d == 'svhn':
                        streams = ['SVHN-CI-1.0']
                    elif d == 'cifar10':
                        streams = ['CIFAR10-CI-1.0']
                    elif d == 'imagenet10':
                        streams, bs = ['IMAGENET10-CI-1.0'], 32
                    elif d == 'cifar20':
                        streams = ['CIFAR20-CI-1.0']
                    elif d == 'imagenet20a':
                        streams, bs = ['IMAGENET20A-CI-1.0'], 32
                    elif d == 'imagenet20b':
                        streams, bs = ['IMAGENET20B-CI-1.0'], 32
                    elif d == 'cifar100p10':
                        streams, ep, n_eval = ['CIFAR100-PRE10-CI-1.0'], 20, 10
                    elif d == 'cifar100p100':
                        streams, ep, n_eval = ['CIFAR100-PRE100-CI-1.0'], 20, 10
                    elif d == 'imagenet200p20':
                        streams, ep, bs, n_eval = ['IMAGENET200-PRE20B-CI-1.0'], 20, 32, 10
                    elif d == 'imagenet200p200':
                        streams, ep, bs, n_eval = ['IMAGENET200-PRE200-CI-1.0'], 20, 32, 10
                    else:
                        raise ValueError(f'Data: {d} unknown/unsupported!')

                    if alg in ['icarl', 'gss', 'si', 'lwf']:
                        evaluator = f'AvaEval'
                    elif alg in ['ersb', 'der']:
                        evaluator = f'IncEval-ep{ep}-bs{bs}'
                    else:
                        evaluator = f'FullIncEval-ep{ep}'

                    base = MIXBaselineExperiment(
                        run_label=f'{run_label}/{alg}',
                        prefix='baselines',
                        device=device,
                        results_dir=f'results/{run_label}/{alg}',
                        lr=lr,
                        bn=bool(bn),
                        epochs=ep,
                        batch_size=bs,
                        n_eval=n_eval,
                        r=123
                    )

                    a = f'{get_alg_mapping(alg, d)}-lr{lr}-bn{bool(bn)}' if alg != 'offline' \
                        else f'{get_alg_mapping(alg, d)}-lr{lr}-ep{ep}'

                    base.run(algorithms=[a], streams=streams, evaluators=[evaluator])
