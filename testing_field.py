import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import argparse
import math

from art.attacks.evasion import FastGradientMethod, ZooAttack, UniversalPerturbation, FeatureAdversariesPyTorch, DeepFool, SquareAttack, Wasserstein
from art.estimators.classification import PyTorchClassifier
from evo_attack import EvoAttack
from runner import get_model
from train import load_dataset

dataset = 'cifar10'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models_names = ['custom', 'densenet121', 'mobilenet_v2', 'resnet18', 'googlenet']
datasets_names = ['cifar10', 'mnist']
metrics = ['l2', 'linf']

def correctly_classified(model, x, y):
    softmax = nn.Softmax(dim=1)
    return y == torch.argmax(softmax(model(x)))

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

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(datasets)

    model = get_model(models, datasets)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.array(y_test)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    count = 0
    success_count = 0
    queries = []
    correctly_classified_images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if correctly_classified(model, x, y) and count < n_images:
            count += 1
            correctly_classified_images_indices.append(i)
            adv, n_queries, success = EvoAttack(model=model, img=x, label=y, metric='linf', delta=delta, perturbed_pixels=1, n_gen=500, pop_size=40,
                            kernel_size=5, min_pixel = min_pixel_value, max_pixel = max_pixel_value).evolve()
            adv = adv.cpu().numpy()
            if i == 0:
                evo_x_test_adv = adv
            else:
                evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
            if success:
                success_count += 1
            queries.append(n_queries)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]


    zoo_attack = ZooAttack(classifier=classifier)
    x_test_adv = zoo_attack.generate(x=x_test)
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print('########################################')
    print(f'Summary:\n')
    print(f'\tModel - {models[0]}\n')
    print(f'\tMetric - {metrics[0]}, delta - {delta:.4f}\n')
    print(f'\tZOO queries: {zoo_attack.queries}\n')
    print("\tAccuracy on Zoo test examples: {:.4f}%\n".format(accuracy * 100))
    print(f'\tMedian of queries - {int(np.median(zoo_attack.queries))}\n')
    print(f'\tAccuracy on Evo test examples: {(1 - success_count / len(y_test)) * 100:.4f}%\n')
    print(f'\tMedian of queries - {int(np.median(n_queries))}\n')
    print('########################################')