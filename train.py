from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy

from model import ConvNet

def load_dataset(name='mnist'):
    if name == 'mnist':
        # transforms.Normalize(# (0.1307,), (0.3081,))
        # train_set = datasets.MNIST('./data', train=True, download=True,
        #                            transform=transforms.Compose([transforms.ToTensor()]))
        test_set = datasets.MNIST('./data', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))

    elif name == 'cifar10':
        # train_set = datasets.CIFAR10('./data', train=True, download=True,
        #                              transform=transforms.Compose([transforms.ToTensor(),
        #                                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                                                                                 (0.2471, 0.2435, 0.2616))]))
        test_set = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                       (0.2471, 0.2435, 0.2616))]))
    elif name == 'imagenet':
        # train_set = datasets.ImageNet('./data', train=True, download=True,
        #                              transform=transforms.Compose([transforms.ToTensor(),
        #                                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
        #
        #                                                                                 (0.2471, 0.2435, 0.2616))]))
        test_set = datasets.ImageNet('/cs_storage/public_datasets/ImageNet', split='val',
                                    transform=transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225))]))
    else:
        Exception('No such dataset')

    # train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=512, shuffle=True)

    # x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    # return (x_train, y_train), (x_test, y_test), next(iter(train_loader))[0].min(), next(iter(train_loader))[0].max()
    # return (None, None), (x_test, y_test), next(iter(train_loader))[0].min(), next(iter(train_loader))[0].max()
    return (x_test, y_test), next(iter(test_loader))[0].min(), next(iter(test_loader))[0].max()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    dataset = 'cifar10'
    in_channels = 3
    out_channels = [32, 64, 128]
    fc_dims = [512, 128, 10]
    batch_size = 128
    n_epochs = 300
    lr = 0.02
    weight_decay = 1e-6
    net = ConvNet(in_channels=in_channels, out_channels=out_channels, fc_dims=fc_dims).to(device)
    train_loader, test_loader = load_dataset(dataset, batch_size=batch_size)
    model, history = train_model(net, n_epochs, train_loader, test_loader, lr, weight_decay)
    torch.save(model, f'{dataset}_model.pth')
