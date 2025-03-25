# -*- coding: utf-8 -*-

import numpy as np
from client import Client
from server import Server
from torchvision import datasets, transforms
import torch
from model import  get_model
from utils import plot
from torch.utils.data import DataLoader, TensorDataset 


def main():
    np.random.seed(42)


    # cifar10_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # cifar10_train_dataset = datasets.cifar10(root='./data', train=True, download=True, transforms=transforms)
    # cifar10_test_dataset = datasets.cifar10(root='./data', train=False, download=True, transforms=transforms)


    train_dataset =  datasets.MNIST('./mnist_dataset', train=True, download=True)
    test_dataset = datasets.MNIST('./mnist_dataset', train=False, download=True)

    train_data = train_dataset.data.to(torch.float)
    train_labels = train_dataset.targets.numpy()

    test_data = test_dataset.data.to(torch.float)
    test_labels = test_dataset.targets.numpy()
    
    
    # mean = (train_data.mean()) / (train_data.max() - train_data.min())
    # std = (train_data.std() / (train_data.max() - train_data.min()))
    minst_tranform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = minst_tranform(train_data)
    test_data = minst_tranform(test_data)
    train_data_size = train_data.shape[0]
    test_data_size = test_data.shape[0]
    test_data_loader = DataLoader(TensorDataset(test_data, torch.as_tensor(test_labels)), batch_size = 64)

    classes = np.max(train_labels) + 1
    clients_train_data = {}
    clients_train_label = {}
    # 划分数据集
    # idxs = np.random.permutation(train_data_size)
    # batch_idxs = np.array_split(idxs, 10)
    alpha = 0.99
    batch_idxs = []
    for i in range(10):
        batch_idxs.append([])
    print(batch_idxs)
    for id in range(10):
        all_idxs = []
        for j, label in enumerate(train_labels):
            if label == id:
                all_idxs.append(j)
        idxs = np.random.permutation(len(all_idxs))
        num = 0
        client_id = 0
        for j in range(len(idxs)):
            if client_id == id:
                if num + 1 <= int(alpha * len(idxs)):
                    batch_idxs[client_id].append(all_idxs[idxs[j]])
                    num += 1
                else:
                    client_id += 1
                    num = 0
            else:
                if (num + 1) * 9 <= len(idxs) - int(alpha  * len(idxs)):
                    batch_idxs[client_id].append(all_idxs[idxs[j]])
                    num += 1
                else:
                    num = 0
                    client_id += 1
        

    for i in range(10):
        clients_train_data[i] = train_data[batch_idxs[i]]
        clients_train_label[i] = train_labels[batch_idxs[i]]
        distribution = [list(clients_train_label[i]).count(j) for j in range(classes)]
        print(distribution)

    
    device = 'cuda'
    clients = []
    for i in range(10):
        clients.append(Client(i, clients_train_data[i], clients_train_label[i]))
    server = Server(rounds=500, 
                    clients=clients, 
                    test_dataloader=test_data_loader,  
                    global_model=get_model().to(device=device))
                    
    accuracy_history = server.train()
    accuracy_history = [acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracy_history]
    # plot(accuracy_history)
if __name__ == '__main__':
    main()
