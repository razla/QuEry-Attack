from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
torch.manual_seed(1)

IAMGENET_PATH = '/cs_storage/public_datasets/ImageNet'

def load_dataset(name='mnist'):
    if name == 'mnist':
        test_set = datasets.MNIST('./data', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), ]))
    elif name == 'cifar10':
        test_set = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),]))
    elif name == 'imagenet':
        test_set = datasets.ImageNet(IAMGENET_PATH, split='val',
                                    transform=transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  ]))
    else:
        Exception('No such dataset')

    test_loader = DataLoader(test_set, batch_size=1024, shuffle=True)
    x_test, y_test = next(iter(test_loader))

    return (x_test, y_test), x_test.min(), x_test.max()
