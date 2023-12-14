import matplotlib.pyplot as plt
import torch
from models.load_pretrained_model import load_resnet
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np


def feedforward(x, resnet):
    return resnet(x).view(64)


def get_representations(resnet, model_name, train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = CIFAR10(root='./CIFAR10', download=True, train=train, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize]))
    representation = torch.zeros((len(dataset), 64))

    print("Processing the data")
    for i in tqdm(range(len(dataset))):
        x = dataset[i][0].view((1, 3, 32, 32))
        with torch.no_grad():
            representation[i] = feedforward(x, resnet)

    df = pd.DataFrame(representation.detach().numpy())
    if train:
        df.to_csv(f'./CIFAR10/representations/{model_name}_train.csv')
    else:
        df.to_csv(f'./CIFAR10/representations/{model_name}_test.csv')
    return df


def main(model):
    resnet = load_resnet(model)
    resnet.eval()

    get_representations(resnet, model, train=True)
    get_representations(resnet, model, train=False)


class CIFAR10Representations(Dataset):
    def __init__(self, csv_file, train, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.representation = pd.read_csv(csv_file)
        self.cifar = CIFAR10(root='./data/CIFAR10', download=True, train=train, transform=transforms.Compose([
            transforms.ToTensor()]))
        self.transform = transform

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        representation = self.representation.iloc[idx, 1:]
        cifar_img = self.cifar[idx][0]
        label = self.cifar[idx][1]

        representation = np.array(representation, dtype=float)
        cifar_img = np.array(cifar_img, dtype=float)

        if self.transform:
            cifar_img = self.transform(cifar_img)

        return representation, cifar_img, label


if __name__ == '__main__':
    model = 'resnet1202'
    # main(model)
    ds = CIFAR10Representations(f'./CIFAR10/representations/{model}_train.csv', train=True)
    print(len(ds))
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    for r, x, y in dl:
        print(r.shape)
        print(x)
        print(y)
        break

