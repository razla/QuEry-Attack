from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import copy
import sys

from models.model import ConvNet
from models.resnet import resnet50
from models.inception import inception_v3
from models.vgg import vgg16_bn
from utils import get_model

import models

models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn']
datasets_names = ['mnist', 'imagenet', 'cifar10', 'svhn']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dataset(dataset):
    if dataset == 'svhn':
        train_set = datasets.SVHN('./data', split='train', download=True,
                      transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_set = datasets.SVHN('./data', split='test', download=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif dataset == 'mnist':
        train_set = datasets.MNIST('./data', train=True, download=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        test_set = datasets.MNIST('./data', train=False, download=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    return train_loader, test_loader


def train_model(net, n_epochs, train_loader, test_loader, lr, weight_decay):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    history = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = -1

    for epoch in range(n_epochs):
        train_losses = []
        train_accs = []
        net = net.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            net.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = output.argmax(axis=1)
            train_losses.append(loss.item())
            train_accs.append((y_pred == labels).sum().item() / len(labels))

        test_losses = []
        test_accs = []
        net = net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                loss = criterion(output, labels)

                y_pred = output.argmax(axis=1)
                test_losses.append(loss.item())
                test_accs.append((y_pred == labels).sum().item() / len(labels))

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        test_loss = np.mean(test_losses)
        test_acc = np.mean(test_accs)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(net.state_dict())

        print(
            f'Epoch #{epoch + 1}:\n\t train loss {train_loss:.4f} train acc {train_acc:.4f}\n\t test loss {test_loss:.4f} test acc {test_acc:.4f}')

    net.load_state_dict(best_model_wts)
    return net.eval(), history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=['ALL'] + models_names, default='custom',
                        help="Run only specific model, or 'ALL' models")
    parser.add_argument("--dataset", "-da", choices=['ALL'] + datasets_names, default='cifar10',
                        help="Run only specific dataset, or 'ALL' datasets")
    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    n_epochs = 200
    lr = 0.01
    weight_decay = 1e-6
    train_loader, test_loader = load_dataset(dataset)
    net = ConvNet(in_channels=1, fc_dims=[128, 64, 10])

    net.to(device)

    model, history = train_model(net, n_epochs, train_loader, test_loader, lr, weight_decay)

    torch.save(model, f'./models/state_dicts/{dataset}_model.pth')