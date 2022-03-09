from art.attacks.evasion import AutoProjectedGradientDescent, SimBA, SquareAttack, CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import numpy as np
import argparse
import random
import torch
import math

from attack import EvoAttack
from utils import get_model
from datasets_loader import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['custom', 'mobilenet_v2', 'googlenet', 'densenet121', 'resnet18']
datasets_names = ['imagenet', 'cifar10', 'svhn']
metrics = ['l2', 'linf']

def add_module(model_name, model):
    if model_name == 'mobilenet_v2':
        model.classifier.add_module('2', torch.nn.Softmax(dim=1))
    elif model_name == 'googlenet':
        model.fc = nn.Sequential(
            model.fc,
            torch.nn.Softmax(dim=1),
        )
    elif model_name == 'resnet18':
        model.fc = nn.Sequential(
            model.fc,
            torch.nn.Softmax(dim=1),
        )
    return model

def correctly_classified(model, x, y):
    softmax = nn.Softmax(dim=1)
    return y == torch.argmax(softmax(model(x)))

def compute_accuracy(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.array(y_test)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    return accuracy

def pgd_attack(init_model, min_ball, max_ball, x_test, i, pgd_adv):
    pgd_classifier = PyTorchClassifier(
        model=init_model,
        clip_values=(min_ball.numpy(), max_ball.numpy()),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    pgd_attack = AutoProjectedGradientDescent(estimator=pgd_classifier, norm='inf', max_iter=100)
    pgd_x_test_adv = pgd_attack.generate(x=x_test[[i]].numpy())
    if i != 0:
        pgd_adv = np.concatenate((pgd_x_test_adv, pgd_adv), axis=0)
    else:
        pgd_adv = pgd_x_test_adv
    return pgd_adv, pgd_attack.queries

def simba_attack(init_model, min_ball, max_ball, x_test, i, simba_adv):
    model = add_module(models, init_model)
    simba_model = model
    simba_classifier = PyTorchClassifier(
        model=simba_model,
        clip_values=(min_ball.numpy(), max_ball.numpy()),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    # Step 6: Generate adversarial test examples
    simba_attack = SimBA(classifier=simba_classifier, max_iter=10000)
    simba_x_test_adv = simba_attack.generate(x=x_test[[i]].numpy())
    if i != 0:
        simba_adv = np.concatenate((simba_x_test_adv, simba_adv), axis=0)
    else:
        simba_adv = simba_x_test_adv
    return simba_adv, simba_attack.queries

def square_attack(init_model, min_ball, max_ball, x_test, i, square_adv):
    square_classifier = PyTorchClassifier(
        model=init_model,
        clip_values=(min_ball.numpy(), max_ball.numpy()),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    square_attack = SquareAttack(estimator=square_classifier, norm='inf', max_iter=10000)
    square_x_test_adv = square_attack.generate(x=x_test[[i]].numpy())
    if i != 0:
        square_adv = np.concatenate((square_x_test_adv, square_adv), axis=0)
    else:
        square_adv = square_x_test_adv
    return square_adv, square_attack.queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=['ALL'] + models_names, default='custom',
                        help="Run only specific model, or 'ALL' models")
    parser.add_argument("--dataset", "-da", choices=['ALL'] + datasets_names, default='cifar10',
                        help="Run only specific dataset, or 'ALL' datasets")
    parser.add_argument("--metric", "-t", choices=['ALL'] + metrics, default='ALL',
                        help="Use only specific metric; or 'ALL' metrics")
    parser.add_argument("--delta", "-de", type=float, default=0.2,
                        help="Constrained optimization problem - delta")
    parser.add_argument("--pop", "-pop", type=int, default=20,
                        help="Population size")
    parser.add_argument("--gen", "-g", type=int, default=10000,
                        help="Number of generations")
    parser.add_argument("--images", "-i", type=int, default=None,
                        help="Maximal number of images from dataset to process (or None, to process all)")
    parser.add_argument("--local", "-l", action='store_true',
                        help='Runs on local machine (default: use Slurm if available)')
    args = parser.parse_args()

    if args.model == 'ALL':
        models = models_names
    else:
        models = args.model

    if args.dataset =='ALL':
        datasets = datasets_names
    else:
        datasets = args.dataset

    if args.metric != 'ALL':
        metrics = [args.metric]

    if args.images is None:
        n_images = math.inf
    else:
        n_images = args.images

    delta = args.delta
    pop_size = args.pop
    n_gen = args.gen

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(datasets)

    init_model = get_model(models, datasets)

    classifier = PyTorchClassifier(
        model=init_model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=nn.CrossEntropyLoss(),
        input_shape=x_test[0].numpy().shape,
        nb_classes=10,
    )

    compute_accuracy(classifier, x_test, y_test)

    count = 0
    success_count = 0
    evo_queries = []
    correctly_classified_images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if count == n_images:
            break

        if correctly_classified(init_model, x, y) and count < n_images:
            count += 1
            correctly_classified_images_indices.append(i)
            adv, n_queries, success = EvoAttack(dataset=datasets, model=init_model, img=x, label=y, metric='linf', delta=delta, n_gen=n_gen, pop_size=pop_size,
                            kernel_size=(x.cpu().numpy().shape[2] // 2) - 1, min_pixel = min_pixel_value, max_pixel = max_pixel_value).evolve()
            adv = adv.cpu().numpy()
            if count == 1:
                evo_x_test_adv = adv
            else:
                evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
            if success:
                success_count += 1
            evo_queries.append(n_queries)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]

    simba_queries = []
    pgd_queries = []
    square_queries = []
    pgd_adv = None
    simba_adv = None
    square_adv = None
    for i in range(len(x_test)):

        min_ball = torch.tile(torch.maximum(x_test[i] - delta, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[i] + delta, max_pixel_value), (1, 1))

        # pgd_adv, pgd_n_queries = pgd_attack(init_model, min_ball, max_ball, x_test, i, pgd_adv)

        simba_adv, simba_n_queries = simba_attack(init_model, min_ball, max_ball, x_test, i, simba_adv)

        square_adv, square_n_queries = square_attack(init_model, min_ball, max_ball, x_test, i, square_adv)

        # pgd_queries.append(pgd_n_queries)
        simba_queries.append(simba_n_queries)
        # carlini_queries.append(carlini_attack.queries)
        square_queries.append(square_n_queries)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    simba_accuracy = compute_accuracy(classifier, simba_adv, y_test)
    # pgd_accuracy = compute_accuracy(classifier, pgd_adv, y_test)
    square_accuracy = compute_accuracy(classifier, square_adv, y_test)

    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {datasets}')
    print(f'\tModel: {models}')
    print(f'\tMetric: {metrics[0]}, delta: {delta:.4f}')
    print(f'\tSimba:')
    print(f'\t\tSimBA - test accuracy: {simba_accuracy * 100:.4f}%')
    print(f'\t\tSimba - queries: {simba_queries}')
    print(f'\t\tSimba - queries (median): {int(np.median(simba_queries))}')
    # print(f'\tPGD:')
    # print(f'\t\tPGD - test accuracy: {pgd_accuracy * 100:.4f}%')
    # print(f'\t\tPGD - queries: {pgd_queries}')
    # print(f'\t\tPGD - queries (median): {int(np.median(pgd_queries))}')
    print(f'\tSquare:')
    print(f'\t\tSquare - test accuracy: {square_accuracy * 100:.4f}%')
    print(f'\t\tSquare - queries: {square_queries}')
    print(f'\t\tSquare - queries (median): {int(np.median(square_queries))}')
    print(f'\tEvo:')
    print(f'\t\tEvo - test accuracy: {(1 - (success_count / n_images)) * 100:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    print('########################################')
