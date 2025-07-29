import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST, FashionMNIST
from data.create_cifar_represenatations import CIFAR10Representations
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(batch_size_tr=128, batch_size_val=128, batch_size_test=128, split_seed=42):
    torch.manual_seed(split_seed)
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    # seed for binomial
    np.random.seed(777)

    # train_samples = np.random.binomial(1, train_dataset.data[:50000] / 255)
    train_samples = train_dataset.data[:50000] / 255
    train_labels = train_dataset.targets[:50000]
    # train = TensorDataset(torch.from_numpy(train_samples), train_labels)
    train = TensorDataset(train_samples, train_labels)
    train_dataloader = DataLoader(train, batch_size=batch_size_tr, shuffle=True, num_workers=4, pin_memory=True)

    val_labels = train_dataset.targets[50000:]
    val_samples = np.random.binomial(1, train_dataset.data[50000:] / 255)
    val = TensorDataset(torch.from_numpy(val_samples), val_labels)
    val_dataloader = DataLoader(val, batch_size=batch_size_val, shuffle=True, num_workers=4, pin_memory=True)

    test_samples = np.random.binomial(1, test_dataset.data / 255)
    test_labels = test_dataset.targets
    test = TensorDataset(torch.from_numpy(test_samples), test_labels)
    test_dataloader = DataLoader(test, batch_size=batch_size_test, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_mnist_small(batch_size_tr=128, batch_size_val=128, batch_size_test=128, split_seed=42):
    torch.manual_seed(split_seed)
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # switch to hogher no
    N = 10

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)

    np.random.seed(777)
    train_samples = train_dataset.data[:N] / 255
    train_labels = train_dataset.targets[:N]

    train = TensorDataset(train_samples, train_labels)
    train_dataloader = DataLoader(train, batch_size=batch_size_tr, shuffle=True, num_workers=4, pin_memory=True)

    val_labels = train_dataset.targets[N:N*2]
    val_samples = np.random.binomial(1, train_dataset.data[N:N*2] / 255)
    val = TensorDataset(torch.from_numpy(val_samples), val_labels)
    val_dataloader = DataLoader(val, batch_size=batch_size_val, shuffle=True, num_workers=4, pin_memory=True)

    test_samples = np.random.binomial(1, test_dataset.data / 255)
    test_labels = test_dataset.targets
    test = TensorDataset(torch.from_numpy(test_samples), test_labels)
    test_dataloader = DataLoader(test, batch_size=batch_size_test, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_fashion_mnist(batch_size_tr=128, batch_size_val=128, batch_size_test=128, split_seed=42):
    torch.manual_seed(split_seed)
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = FashionMNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    test_dataset = FashionMNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)

    # seed for binomial
    np.random.seed(777)

    # train_samples = np.random.binomial(1, train_dataset.data[:50000] / 255)
    train_samples = train_dataset.data[:50000] / 255
    train_labels = train_dataset.targets[:50000]
    # train = TensorDataset(torch.from_numpy(train_samples), train_labels)
    train = TensorDataset(train_samples, train_labels)
    train_dataloader = DataLoader(train, batch_size=batch_size_tr, shuffle=True, num_workers=4, pin_memory=True)

    val_labels = train_dataset.targets[50000:]
    val_samples = np.random.binomial(1, train_dataset.data[50000:] / 255)
    val = TensorDataset(torch.from_numpy(val_samples), val_labels)
    val_dataloader = DataLoader(val, batch_size=batch_size_val, shuffle=True, num_workers=4, pin_memory=True)

    test_samples = np.random.binomial(1, test_dataset.data / 255)
    test_labels = test_dataset.targets
    test = TensorDataset(torch.from_numpy(test_samples), test_labels)
    test_dataloader = DataLoader(test, batch_size=batch_size_test, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_CIFAR10(batch_size_tr=128, batch_size_val=128, batch_size_test=128,
                 split_seed=42, resnet_model='resnet1202'):

    torch.manual_seed(split_seed)

    train_dataset = CIFAR10Representations(f'./data/CIFAR10/representations/{resnet_model}_train.csv', train=True)
    test_dataset = CIFAR10Representations(f'./data/CIFAR10/representations/{resnet_model}_test.csv', train=False)

    np.random.seed(777)

    train_set, val_set = torch.utils.data.random_split(train_dataset, [40000, 10000])

    train_dataloader = DataLoader(train_set, batch_size=batch_size_tr, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_CIFAR10_small(batch_size_tr=128, batch_size_val=128, batch_size_test=128,
                 split_seed=42, resnet_model='resnet1202'):

    torch.manual_seed(split_seed)

    train_dataset = CIFAR10Representations(f'./data/CIFAR10/representations/{resnet_model}_train.csv', train=True)
    test_dataset = CIFAR10Representations(f'./data/CIFAR10/representations/{resnet_model}_test.csv', train=False)

    np.random.seed(777)

    train_set, val_set = torch.utils.data.random_split(train_dataset, [40000, 10000])
    train_set, _ = torch.utils.data.random_split(train_set, [10, 40000-10])
    val_set, _ = torch.utils.data.random_split(val_set, [10, 10000-10])
    test_set, _ = torch.utils.data.random_split(test_dataset, [10, len(test_dataset)-10])
    train_dataloader = DataLoader(train_set, batch_size=batch_size_tr, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
