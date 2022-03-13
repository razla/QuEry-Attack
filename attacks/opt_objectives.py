from art.attacks.evasion.square_attack import SquareAttack
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import torch

from attack import EvoAttack
from utils import get_model
from datasets_loader import load_dataset
from testing_field import correctly_classified, compute_accuracy

datasets_names = ['imagenet', 'cifar10', 'mnist', 'svhn']
models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn']
parser = argparse.ArgumentParser(
    description="Runs hyperparameter optimization on various attacks")
parser.add_argument("--model", "-m", choices=models_names, default='inception_v3',
                    help="Run only specific model")
parser.add_argument("--dataset", "-d", choices=['ALL'] + datasets_names, default='cifar10',
                    help="Run only specific dataset, or 'ALL' datasets")
parser.add_argument("--images", "-i", type=int, default=20,
                    help="Number of images")
parser.add_argument("--delta", "-de", type=float, default=0.4,
                    help="Perturbation")
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset
n_images = args.images
delta = args.delta

device = 'cuda' if torch.cuda.is_available() else 'cpu'

(x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset_name)

def evo_objective(trial):
    global x_test, y_test, model_name, dataset_name

    d_low = trial.suggest_float('d_low', .0001, 1)
    d_high = trial.suggest_float('d_high', .0001, 1)

    init_model = get_model(model_name, dataset_name)

    compute_accuracy(dataset_name, init_model, x_test, y_test, min_pixel_value, max_pixel_value, to_tensor=False)

    count = 0
    success_count = 0
    queries = []
    correctly_classified_images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if count == n_images:
            break

        if correctly_classified(init_model, x, y) and count < n_images:
            count += 1
            correctly_classified_images_indices.append(i)
            adv, n_queries, success = EvoAttack(dataset=dataset_name, model=init_model, img=x, label=y, delta=delta,
                                                kernel_size=(x.cpu().numpy().shape[2] // 2) - 1, n_gen=500, pop_size=40,
                                                n_tournament=2, steps=100, min_pixel=min_pixel_value,
                                                max_pixel=max_pixel_value, d_low=d_low, d_high=d_high).evolve()
            print(f'Queries: {n_queries}')
            adv = adv.cpu().numpy()
            if count == 1:
                evo_x_test_adv = adv
            else:
                evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
            if success:
                success_count += 1
            queries.append(n_queries)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]

    ASR = (1 - (success_count / len(y_test)))
    queries = np.median(n_queries) / np.sum(n_queries)

    print(f'ASR: {ASR:.4f}, queries: {queries:.4f}')
    return queries + ASR