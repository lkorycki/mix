import torch
import wget

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

from data.cifar100_coarse import CIFAR100Coarse
from data.data_utils import DataUtils
from data.post_funcs import imagenet200_val_post
from data.svhn2 import SVHN2
from data.tensor_set import TensorDataset

pytorch_data_root = './pytorch_data'
arff_data_root = './arff_data'


data_creators = {
    'MNIST-TRAIN': lambda: datasets.MNIST(pytorch_data_root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
    'MNIST-TEST': lambda: datasets.MNIST(pytorch_data_root, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
    'MNIST-TRAIN-FLAT': lambda: datasets.MNIST(pytorch_data_root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])),
    'MNIST-TEST-FLAT': lambda: datasets.MNIST(pytorch_data_root, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])),
    'MNIST-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/mnist10-train.pt'),
    'MNIST-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/mnist10-test.pt'),
    'FASHION-TRAIN': lambda: datasets.FashionMNIST(pytorch_data_root, train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
    'FASHION-TEST': lambda: datasets.FashionMNIST(pytorch_data_root, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
    'FASHION-TRAIN-FLAT': lambda: datasets.FashionMNIST(pytorch_data_root, train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])),
    'FASHION-TEST-FLAT': lambda: datasets.FashionMNIST(pytorch_data_root, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])),
    'FASHION-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/fashion10-train.pt'),
    'FASHION-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/fashion10-test.pt'),
    'SVHN-TRAIN': lambda: SVHN2(pytorch_data_root, split='train', transform=transforms.Compose([transforms.ToTensor()]), download=True),
    'SVHN-TEST': lambda: SVHN2(pytorch_data_root, split='test', transform=transforms.Compose([transforms.ToTensor()]), download=True),
    'SVHN-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/svhn10-train.pt'),
    'SVHN-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/svhn10-test.pt'),
    'CIFAR10-TRAIN': lambda: datasets.CIFAR10(pytorch_data_root, True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]), download=True),
    'CIFAR10-TEST': lambda: datasets.CIFAR10(pytorch_data_root, False, transform=transforms.Compose([transforms.ToTensor()]), download=True),
    'CIFAR10-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar10-train.pt'),
    'CIFAR10-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar10-test.pt'),
    'CIFAR100-TRAIN': lambda: datasets.CIFAR100(pytorch_data_root, True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]), download=True),
    'CIFAR100-TEST': lambda: datasets.CIFAR100(pytorch_data_root, False, transform=transforms.Compose([transforms.ToTensor()]), download=True),
    'CIFAR100-PRE10-TRAIN': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar100p10-train.pt'),
    'CIFAR100-PRE10-TEST': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar100p10-test.pt'),
    'CIFAR100-PRE100-TRAIN': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar100p100-train.pt'),
    'CIFAR100-PRE100-TEST': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar100p100-test.pt'),
    'CIFAR20C-TRAIN': lambda: CIFAR100Coarse(pytorch_data_root, train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]), download=True),
    'CIFAR20C-TEST': lambda: CIFAR100Coarse(pytorch_data_root, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True),
    'CIFAR20C-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar20c-train.pt'),
    'CIFAR20C-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/cifar20c-test.pt'),
    'IMAGENET200-TRAIN': lambda: DataUtils.create_dataset(pytorch_data_root, 'tiny-imagenet-200/train',
                                                          transform=transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()]),
                                                          download=True, download_func=lambda: wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip', pytorch_data_root),
                                                          zip_file='tiny-imagenet-200.zip', ),
    'IMAGENET200-TEST': lambda: DataUtils.create_dataset(pytorch_data_root, 'tiny-imagenet-200/val',
                                                         transform=transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()]),
                                                         download=True, download_func=lambda: wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip', pytorch_data_root),
                                                         zip_file='tiny-imagenet-200.zip', post_func=imagenet200_val_post),
    'IMAGENET10-TRAIN': lambda: DataUtils.create_dataset_subset(get('IMAGENET200-TRAIN'), [0, 22, 25, 68, 117, 145, 153, 176, 188, 198], f'{pytorch_data_root}/imagenet10-train-indices.pt'),
    'IMAGENET10-TEST': lambda: DataUtils.create_dataset_subset(get('IMAGENET200-TEST'), [0, 22, 25, 68, 117, 145, 153, 176, 188, 198], f'{pytorch_data_root}/imagenet10-test-indices.pt'),
    'IMAGENET10-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet10-train.pt'),
    'IMAGENET10-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet10-test.pt'),
    'IMAGENET20A-TRAIN': lambda: DataUtils.create_dataset_subset(get('IMAGENET200-TRAIN'), [i * 10 - 1 for i in range(1, 21)], f'{pytorch_data_root}/imagenet20a-train-indices.pt'),
    'IMAGENET20A-TEST': lambda: DataUtils.create_dataset_subset(get('IMAGENET200-TEST'), [i * 10 - 1 for i in range(1, 21)], f'{pytorch_data_root}/imagenet20a-test-indices.pt'),
    'IMAGENET20A-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet20a-train.pt'),
    'IMAGENET20A-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet20a-test.pt'),
    'IMAGENET20B-TRAIN': lambda: DataUtils.create_dataset_subset(get('IMAGENET200-TRAIN'), [i * 10 - 5 for i in range(1, 21)], f'{pytorch_data_root}/imagenet20b-train-indices.pt'),
    'IMAGENET20B-TEST': lambda: DataUtils.create_dataset_subset(get('IMAGENET200-TEST'), [i * 10 - 5 for i in range(1, 21)], f'{pytorch_data_root}/imagenet20b-test-indices.pt'),
    'IMAGENET20B-TRAIN-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet20b-train.pt'),
    'IMAGENET20B-TEST-TENSOR': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet20b-test.pt'),
    'IMAGENET200-PRE20B-TRAIN': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet200p20b-train.pt'),
    'IMAGENET200-PRE20B-TEST': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet200p20b-test.pt'),
    'IMAGENET200-PRE1000-TRAIN': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet200p1000-train.pt'),
    'IMAGENET200-PRE1000-TEST': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet200p1000-test.pt'),
    'IMAGENET200-PRE200-TRAIN': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet200p200-train.pt'),
    'IMAGENET200-PRE200-TEST': lambda: TensorDataset(f'{pytorch_data_root}/extracted/imagenet200p200-test.pt'),
    'CELEB-TRAIN': lambda: datasets.CelebA(pytorch_data_root, split='train', target_type='identity', transform=transforms.Compose([transforms.ToTensor()]), download=True),
    'CELEB-TEST': lambda: datasets.CelebA(pytorch_data_root, split='test', target_type='identity', transform=transforms.Compose([transforms.ToTensor()]), download=True)

    # TODO:
    # CORE50: https://vlomonaco.github.io/core50/index.html#dataset
    # IMAGENET1000 (64): https://patrykchrabaszcz.github.io/Imagenet32/

    # A) MNIST, FASHION, SVHN, CIFAR10, IMGNET10, CIFAR20, IMGNET20
    # B) CORE50, CIFAR100, IMGNET200
    # C) CELEB, IMG1000
}


def get(name: str):
    return data_creators[name]()
