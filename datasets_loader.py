from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(name='mnist'):
    if name == 'mnist':
        test_set = datasets.MNIST('./data', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize( (0.1307,), (0.3081,))]))
    elif name == 'cifar10':
        test_set = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                       (0.2471, 0.2435, 0.2616))]))
    elif name == 'imagenet':
        test_set = datasets.ImageNet('/cs_storage/public_datasets/ImageNet', split='val',
                                    transform=transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225))]))
        test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
        x_test, y_test = next(iter(test_loader))
        return (x_test, y_test), x_test.min(), x_test.max()
    elif name == 'svhn':
        test_set = datasets.SVHN('./data', split='test', download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        Exception('No such dataset')

    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    x_test, y_test = next(iter(test_loader))

    return (x_test, y_test), x_test.min(), x_test.max()