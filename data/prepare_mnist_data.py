import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset
import os

def prepare_mnist_data():
    """
    Pre-processes the MNIST dataset by binarizing and splitting it into training,
    validation, and test sets. The processed datasets are saved as .pt files.
    """
    print("Preparing MNIST data...")
    
    # Ensure the data directory exists
    data_dir = './data/MNIST_processed'
    os.makedirs(data_dir, exist_ok=True)

    # Check if the data has already been prepared
    if os.path.exists(os.path.join(data_dir, 'train.pt')) and \
       os.path.exists(os.path.join(data_dir, 'val.pt')) and \
       os.path.exists(os.path.join(data_dir, 'test.pt')):
        print("Pre-processed MNIST data already exists.")
        return

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download and load the raw MNIST data
    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)

    # Set a fixed seed for reproducibility
    np.random.seed(777)

    # Process and save the training set
    train_samples = train_dataset.data[:50000] / 255
    train_labels = train_dataset.targets[:50000]
    train_set = TensorDataset(train_samples, train_labels)
    torch.save(train_set, os.path.join(data_dir, 'train.pt'))

    # Process and save the validation set
    val_samples = np.random.binomial(1, train_dataset.data[50000:] / 255)
    val_labels = train_dataset.targets[50000:]
    val_set = TensorDataset(torch.from_numpy(val_samples), val_labels)
    torch.save(val_set, os.path.join(data_dir, 'val.pt'))

    # Process and save the test set
    test_samples = np.random.binomial(1, test_dataset.data / 255)
    test_labels = test_dataset.targets
    test_set = TensorDataset(torch.from_numpy(test_samples), test_labels)
    torch.save(test_set, os.path.join(data_dir, 'test.pt'))

    print("Finished preparing MNIST data.")

if __name__ == '__main__':
    prepare_mnist_data()
