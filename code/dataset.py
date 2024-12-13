import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor, Compose, Grayscale, Resize

def load_mnist():
    
    transform = Compose([
        Grayscale(num_output_channels=1),
        Resize((28, 28)),
        ToTensor(),
    ])
    
    trainset = MNIST(root='./data', train=True,
                       download=True, transform=transform)
    testset = MNIST(root='./data', train=False,
                      download=True, transform=transform)
    return trainset, testset


def load_cifar10():
    
    transform = Compose([
        Grayscale(num_output_channels=1),
        Resize((28, 28)),
        ToTensor(),
    ])
    
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform)
    return trainset, testset


def load_fashion_mnist():
    
    transform = Compose([
        Grayscale(num_output_channels=1),
        Resize((28, 28)),
        ToTensor(),
    ])
    
    trainset = FashionMNIST(root='./data', train=True,
                       download=True, transform=transform)
    testset = FashionMNIST(root='./data', train=False,
                      download=True, transform=transform)
    return trainset, testset


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
