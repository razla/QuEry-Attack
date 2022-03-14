import numpy as np
import argparse
import torch

from attack import EvoAttack
from utils import get_model
from datasets_loader import load_dataset
from testing_field import correctly_classified, compute_accuracy

MODEL_PATH = '../../models/state_dicts'
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
parser.add_argument("--delta", "-de", type=float, default=0.05,
                    help="Perturbation")
parser.add_argument("--gen", "-g", type=int, default=40,
                    help="Number of generations")
parser.add_argument("--pop", "-p", type=int, default=500,
                    help="Size of population")
parser.add_argument("--d_low", "-dl", type=float, default=1e-5,
                    help="Low diversity threshold")
parser.add_argument("--d_high", "-dh", type=float, default=1e-3,
                    help="High diversity threshold")
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset
n_images = args.images
delta = args.delta
gen = args.gen
pop = args.pop
d_low_sugg = args.d_low
d_high_sugg = args.d_high

device = 'cuda' if torch.cuda.is_available() else 'cpu'

(x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset_name)

def evo_objective(trial):
    global x_test, y_test, model_name, dataset_name

    d_low = trial.suggest_loguniform('d_low', d_low_sugg, 5e-1)
    d_high = trial.suggest_loguniform('d_high', d_high_sugg, 5e-1)

    init_model = get_model(model_name, dataset_name, MODEL_PATH)

    compute_accuracy(dataset_name, init_model, x_test, y_test, min_pixel_value, max_pixel_value, to_tensor=False, to_normalize=True)

    count = 0
    success_count = 0
    queries = []
    correctly_classified_images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if count == n_images:
            break

        if correctly_classified(dataset_name, init_model, x, y) and count < n_images:
            count += 1
            correctly_classified_images_indices.append(i)
            adv, n_queries, success = EvoAttack(dataset=dataset_name,
                                                model=init_model,
                                                img=x,
                                                label=y,
                                                delta=delta,
                                                n_gen=gen,
                                                pop_size=pop,
                                                kernel_size=(x.cpu().numpy().shape[2] // 2) - 1,
                                                min_pixel=min_pixel_value,
                                                max_pixel=max_pixel_value,
                                                d_low=d_low,
                                                d_high=d_high).evolve()
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
    QUERIES = np.mean(queries) / np.max(queries)

    print(f'ASR: {ASR:.4f}, queries: {QUERIES:.4f}')
    return QUERIES + ASR