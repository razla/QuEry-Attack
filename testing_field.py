from art.attacks.evasion import ZooAttack, SimBA
from art.estimators.classification import PyTorchClassifier
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import random
import torch
import math
import wandb

from attack import EvoAttack
from runner import get_model
from train import load_dataset

# wandb.init(project="EvoAttack", entity="razla")

dataset = 'cifar10'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models_names = ['custom', 'mobilenet_v2', 'googlenet']
datasets_names = ['imagenet', 'cifar10', 'mnist']
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
    parser.add_argument("--pixels", "-p", type=int, default=1,
                        help="Number of perturbed pixels per mutation")
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
    pixels = args.pixels
    pop_size = args.pop
    n_gen = args.gen

    # wandb.config = {
    #     "delta": delta,
    #     "perturbed_pixels": pixels
    # }

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(datasets)

    model = get_model(models, datasets)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
            y_list = [label for label in range(10) if torch.tensor(label).to(device) != y]
            random_y = torch.tensor(random.choice(y_list))
            count += 1
            correctly_classified_images_indices.append(i)
            adv, n_queries, success = EvoAttack(model=model, img=x, label=y, targeted_label=random_y, metric='linf', delta=delta, perturbed_pixels=pixels, n_gen=n_gen, pop_size=pop_size,
                            kernel_size=3, min_pixel = min_pixel_value, max_pixel = max_pixel_value).evolve()
            # adv, n_queries, success = EvoAttack(model=model, img=x, label=y, metric='linf', delta=delta, perturbed_pixels=pixels, n_gen=n_gen, pop_size=pop_size,
            #                 kernel_size=5, min_pixel = min_pixel_value, max_pixel = max_pixel_value).evolve()
            adv = adv.cpu().numpy()
            if i == 0:
                evo_x_test_adv = adv
            else:
                evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
            if success:
                success_count += 1
            queries.append(n_queries)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]
    model.classifier.add_module('2', torch.nn.Softmax(dim=1))
    simba_queries = []
    for i in range(len(x_test)):

        min_ball = torch.tile(torch.maximum(x_test[i] - delta, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[i] + delta, max_pixel_value), (1, 1))

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_ball.numpy(), max_ball.numpy()),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )

        # Step 6: Generate adversarial test examples
        attack = SimBA(classifier=classifier)
        x_test_adv = attack.generate(x=x_test[[i]].numpy())
        if i != 0:
            adv = np.concatenate((x_test_adv, x_test_adv), axis=0)
        else:
            adv = x_test_adv
        simba_queries.append(attack.queries)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(adv)
    accuracy = np.sum(predictions == y_test[:50]) / len(y_test[:50])
    print("Accuracy on SimBA test examples: {}%".format(accuracy * 100))

    print('########################################')
    print(f'Summary:')
    print(f'\tModel: {models}')
    print(f'\tMetric: {metrics[0]}, delta: {delta:.4f}')
    print(f'\tPerturbed pixels: {pixels}')
    print(f'\t\tSimBA - test accuracy: {accuracy * 100:.4f}%')
    print(f'\t\tSimba - queries (median): {int(np.median(simba_queries))}')
    print(f'\t\tEvo - test accuracy: {(1 - success_count / len(y_test)) * 100:.4f}%')
    print(f'\t\tEvo - queries (median): {int(np.median(n_queries))}')
    print('########################################')

    # wandb.log({"Queries": int(np.median(n_queries))})
    # wandb.log({"ASR": (1 - success_count / len(y_test)) * 100})