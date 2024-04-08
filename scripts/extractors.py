import torch
import torchvision
from torchsummary import summary
from torch import nn

from data.tensor_set import extract_features
from learners.models.resnet import cifar_resnet
from learners.models.resnext import create_cifar_resnext
from learners.nnet import mnistnet, cifar10_resnet
import data.data_collection as data_col


def extract(last):
    print('Running')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    extractor = mnistnet('pytorch_models/fashion2.pth', device)
    extractor.fc2 = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('FASHION-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/fashion10-train.pt', device=device)
    dataset = data_col.get('FASHION-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/fashion10-test.pt', device=device)

    extractor = mnistnet(f'pytorch_models/mnist2.pth', device)
    extractor.fc2 = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('MNIST-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/mnist10-train.pt', device=device)
    dataset = data_col.get('MNIST-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/mnist10-test.pt', device=device)

    extractor = create_cifar_resnext('pytorch_models/cifar_resnext29.pth.tar')
    extractor.classifier = torch.nn.Identity()
    extractor.eval().to(device)
    summary(extractor.to(device), (3, 32, 32))
    dataset = data_col.get('SVHN-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/svhn10-2-train.pt', device=device)
    dataset = data_col.get('SVHN-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/svhn10-2-test.pt', device=device)

    extractor = cifar10_resnet('pytorch_models/cifar10-2.pth', device)
    extractor.fc2 = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('CIFAR10-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar10-train.pt', device=device)
    dataset = data_col.get('CIFAR10-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar10-test.pt', device=device)

    extractor = create_cifar_resnext('pytorch_models/cifar_resnext29.pth.tar')
    extractor.classifier = torch.nn.Identity()
    extractor.eval().to(device)
    summary(extractor.to(device), (3, 32, 32))
    dataset = data_col.get('CIFAR20C-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar20c-train.pt', device=device)
    dataset = data_col.get('CIFAR20C-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar20c-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet10-2.pth'))
    extractor.fc = extractor.fc[:-1]
    extractor.eval().to(device)
    summary(extractor.to(device), (3, 64, 64))
    dataset = data_col.get('IMAGENET10-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet10-2-train.pt', device=device)
    dataset = data_col.get('IMAGENET10-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet10-2-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet20a-2f.pth'))
    extractor.fc = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET20A-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20a-train.pt', device=device)
    dataset = data_col.get('IMAGENET20A-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20a-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet20b-2f.pth'))
    extractor.fc = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET20B-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20b-train.pt', device=device)
    dataset = data_col.get('IMAGENET20B-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20b-test.pt', device=device)


def extract_long():
    print('Running')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    extractor = cifar10_resnet('pytorch_models/cifar10-2.pth', device)
    extractor.fc2 = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('CIFAR100-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar100p10-train.pt', device=device)
    dataset = data_col.get('CIFAR100-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar100p10-test.pt', device=device)

    extractor = create_cifar_resnext('pytorch_models/cifar_resnext29.pth.tar')
    extractor.classifier = torch.nn.Identity()
    extractor.eval().to(device)
    summary(extractor.to(device), (3, 32, 32))
    dataset = data_col.get('CIFAR100-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar100p100-train.pt', device=device)
    dataset = data_col.get('CIFAR100-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar100p100-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet20b-2f.pth'))
    extractor.fc = extractor.fc[:2]
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET200-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet200p20b-train.pt', device=device)
    dataset = data_col.get('IMAGENET200-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet200p20b-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    extractor.fc = nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET200-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet200p1000-train.pt', device=device)
    dataset = data_col.get('IMAGENET200-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet200p1000-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = nn.Sequential(fc1, nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 200))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet200-2f.pth'))
    extractor.fc = extractor.fc[:2]
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET200-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet200p200-train.pt', device=device)
    dataset = data_col.get('IMAGENET200-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet200p200-test.pt', device=device)


def run():
    extract_long()


if __name__ == '__main__':
    run()
