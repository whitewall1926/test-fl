# -*- coding: utf-8 -*-

from client import Client
from server import Server
from torchvision import datasets, transforms
def main():
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mnist_dataset = datasets.cifar10(root='./data', train=True, download=True, transforms=transforms)
    
    

if __name__ == '__main__':
    main()
