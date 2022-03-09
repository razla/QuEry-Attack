from torchvision.utils import save_image
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
import torch

from models.googlenet import googlenet
from models.densenet import densenet121, densenet169
from models.resnet import resnet18
from models.mobilenetv2 import mobilenet_v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

def get_model(model_name, dataset):
    if model_name == 'custom':
        return torch.load(Path('models/state_dicts') / f'{dataset}_model.pth', map_location=torch.device(device))
    elif dataset=='cifar10':
        return globals()[model_name](pretrained=True).to(device)
    elif dataset=='imagenet':
        return models.mobilenet_v2(pretrained=True).to(device)
    elif dataset=='svhn':
        return torch.load(Path('models/state_dicts') / f'{dataset}_{model_name}_model.pth', map_location=torch.device(device))
    else:
        raise Exception('No such dataset!')

def inv_normalize(dataset):
    if dataset == 'cifar10':
        return transforms.Normalize(
                mean=[-0.4914 / 0.2471, -0.4822 / 0.2435, -0.4465 / 0.2616],
                std=[1 / 0.2471, 1 / 0.2435, 1 / 0.2616]
            )
    elif dataset == 'mnist':
        return transforms.Normalize(mean = [-0.1307 / 0.3081], std=[1 / 0.3081])

    elif dataset == 'svhn':
        return transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
    elif dataset == 'imagenet':
        return transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
    else:
        return 'WTF'

def inv_normalize_and_save(dataset, img, best_individual, not_best_individual):
    orig = inv_normalize(dataset)(img)
    if (not_best_individual != None and best_individual != None):
        good_attack = inv_normalize(dataset)(best_individual)
        bad_attack = inv_normalize(dataset)(not_best_individual)
        save_image(good_attack, 'good.png')
        save_image(bad_attack, 'bad.png')
        save_image(orig, 'orig.png')

def print_initialize(softmax, model, img, label):
    print("################################")
    print(f'Correct class: {label}')
    print(f'Initial class prediction: {model(img).argmax(dim=1).item()}')
    print(f'Initial probability: {softmax(model(img)).max():.4f}')
    print("################################")

def print_success(softmax, model, img, n_queries, label, best_individual, gen):
    print("################################")
    print(f'Evolution succeeded in gen #{gen + 1}')
    print(f'Correct class: {label}')
    print(f'Current prediction: {model(best_individual).argmax(dim=1).item()}')
    print(
        f'Current probability (orig class): {softmax(model(best_individual))[0][label].item():.4f}')
    l_infinity = torch.norm(img - best_individual, p=float('inf')).item()
    print(f'L infinity: {l_infinity:.4f}')
    print(f'Number of queries: {n_queries}')
    print("################################")

def print_failure(softmax, model, img, n_queries, label, best_individual, gen):
    print("################################")
    print("Evolution failed")
    print(f'Correct class: {label}')
    print(f'Current prediction: {model(best_individual).argmax(dim=1).item()}')
    print(
        f'Current probability (orig class): {softmax(model(best_individual))[0][label].item():.4f}')
    print("################################")