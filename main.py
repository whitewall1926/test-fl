# -*- coding: utf-8 -*-

import numpy as np
from client import Client
from server import Server
from torchvision import datasets, transforms
import torch
from model import  get_model
from utils import (
    plot, 
    create_long_tail_split_noniid, 
    create_dirichlet_split_noniid,
    load_config
)
from torch.utils.data import DataLoader, TensorDataset 
import json
import os


def main():
    config = load_config()
    save_path = 'log_data/config.json'
    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    np.random.seed(config['system']['seed'])


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
    test_data_loader = DataLoader(TensorDataset(test_data, 
                                                torch.as_tensor(test_labels)), 
                                                batch_size = config['train']['batch_size'])

    
    # clients_train_data, clients_train_label = create_long_tail_split_noniid(train_data=train_data,
    #                                                                     train_labels=train_labels,
    #                                                                     alpha=0.1,
    #                                                                     clients_number=10)
    clients_train_data, clients_train_label = create_dirichlet_split_noniid(
        train_data=train_data,
        train_labels=train_labels,
        alpha=config['dataset']['alpha'],
        clients_number=config['train']['num_clients'],
        seed=config['system']['seed']  # dataset productivity
    )
    
    device = config['system']['device']
    clients = []
    model = get_model()
    for i in range(10):
        clients.append(Client(i, clients_train_data[i], clients_train_label[i]))
    server = Server(rounds=config['train']['global_rounds'], 
                    clients=clients, 
                    test_dataloader=test_data_loader,  
                    global_model=model.to(device=device))
                    
    accuracy_history = server.train(beta=config['strategy']['iwds']['beta'],
                                    beta_zero=config['strategy']['iwds']['beta_zero'],
                                    rou=config['strategy']['iwds']['rou'],
                                    log_dir=config['system']['log_dir'],
                                    local_epoch=config['train']['local_epoch'],
                                    local_batch_size=config['train']['batch_size'],
                                    learning_rate=config['optimizer']['learning_rate'],
                                    limit_rounds=config['strategy']['client_gini']['limit_rounds'])
    
    accuracy_history = [acc.cpu().item() 
                        if isinstance(acc, torch.Tensor) else acc 
                        for acc in accuracy_history]
    
if __name__ == '__main__':
    main()
