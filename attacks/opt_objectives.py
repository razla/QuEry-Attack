from art.attacks.evasion.simba import SimBA
from art.attacks.evasion.zoo import ZooAttack
from art.attacks.evasion.square_attack import SquareAttack
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import torch

from attack import EvoAttack
from runner import get_model
from train import load_dataset

datasets_names = ['imagenet', 'cifar10', 'mnist']
parser = argparse.ArgumentParser(
    description="Runs hyperparameter optimization on various attacks")
parser.add_argument("--dataset", "-d", choices=['ALL'] + datasets_names, default='cifar10',
                    help="Run only specific dataset, or 'ALL' datasets")
parser.add_argument("--images", "-i", type=int, default=20,
                    help="Number of images")
args = parser.parse_args()
dataset_name = args.dataset
n_images = args.images

device = 'cuda' if torch.cuda.is_available() else 'cpu'

(x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset_name)

def correctly_classified(model, x, y):
    softmax = nn.Softmax(dim=1)
    return y == torch.argmax(softmax(model(x)))

def compute_accuracy(classifier):
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.array(y_test)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

def square_objective(trial):
    global x_test, y_test
    p_init = trial.suggest_float('p_init', .1, 1)
    nb_restarts = trial.suggest_int('nb_restarts', 1, 20)

    model = get_model('mobilenet_v2', 'cifar10')

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

    compute_accuracy(classifier)

    count = 0
    correctly_classified_images_indices = []

    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if correctly_classified(model, x, y) and count < n_images:
            count += 1
            correctly_classified_images_indices.append(i)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]

    square_queries = []
    for i in range(len(x_test)):
        min_ball = torch.tile(torch.maximum(x_test[[i]] - 0.4, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[[i]] + 0.4, max_pixel_value), (1, 1))

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_ball.numpy(), max_ball.numpy()),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )

        # Step 6: Generate adversarial test examples
        attack = SquareAttack(estimator=classifier,
                           max_iter=10000,
                           norm='inf',
                           eps=0.4,
                           p_init=p_init,
                           nb_restarts=nb_restarts,
                           batch_size=1,
                           verbose=False)
        x_test_adv = attack.generate(x=x_test[[i]].numpy())
        if i != 0:
            adv = np.concatenate((x_test_adv, x_test_adv), axis=0)
        else:
            adv = x_test_adv
        print(f'Queries: {attack.queries}')
        square_queries.append(attack.queries)

    predictions = classifier.predict(adv)
    ASR = np.sum(predictions == y_test[:n_images]) / len(y_test[:n_images])
    queries = np.median(square_queries) / np.sum(square_queries)
    print(f'ASR: {ASR:.4f}, queries: {queries:.4f}')
    return queries + ASR

def zoo_objective(trial):
    global x_test, y_test
    learning_rate = trial.suggest_float('epsilon', .01, 1)
    binary_search_steps = trial.suggest_int('binary_search_Steps', 1, 100)
    variable_h = trial.suggest_float('variable_h', .001, 1)

    model = get_model('mobilenet_v2', 'cifar10')

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

    compute_accuracy(classifier)

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

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]

    # model.classifier.add_module('2', torch.nn.Softmax(dim=1))
    n_queries = []
    for i in range(len(x_test)):
        min_ball = torch.tile(torch.maximum(x_test[[i]] - 0.4, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[[i]] + 0.4, max_pixel_value), (1, 1))

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_ball.numpy(), max_ball.numpy()),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )

        # Step 6: Generate adversarial test examples
        attack = ZooAttack(classifier=classifier,
                           max_iter=10000,
                           # learning_rate=learning_rate,
                           batch_size=1)
                           # binary_search_steps=binary_search_steps,
                           # variable_h=variable_h)
        x_test_adv = attack.generate(x=x_test[[i]].numpy())
        if i != 0:
            adv = np.concatenate((x_test_adv, x_test_adv), axis=0)
        else:
            adv = x_test_adv
        n_queries.append(attack.queries)

    predictions = classifier.predict(adv)
    ASR = np.sum(predictions == y_test[:n_images]) / len(y_test[:n_images])
    queries = np.median(n_queries) / np.sum(n_queries)
    print(f'ASR: {ASR:.4f}, queries: {queries:.4f}')
    return queries + ASR

def simba_objective(trial):
    global x_test, y_test
    # attack = trial.suggest_categorical('attack', ['px', 'dct'])
    epsilon = trial.suggest_float('epsilon', .001, 1)
    # freq_dim = trial.suggest_int('freq_dim', 32, 100)

    model = get_model('mobilenet_v2', 'cifar10')

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

    compute_accuracy(classifier)

    count = 0
    success_count = 0
    n_queries = []
    correctly_classified_images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if correctly_classified(model, x, y) and count < n_images:
            count += 1
            correctly_classified_images_indices.append(i)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]

    model.classifier.add_module('2', torch.nn.Softmax(dim=1))
    n_queries = []
    for i in range(len(x_test)):

        min_ball = torch.tile(torch.maximum(x_test[i] - 0.4, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[i] + 0.4, max_pixel_value), (1, 1))

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_ball.numpy(), max_ball.numpy()),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )

        # Step 6: Generate adversarial test examples
        attack = SimBA(classifier=classifier, max_iter=10000, epsilon=epsilon)
        x_test_adv = attack.generate(x=x_test[[i]].numpy())
        if i != 0:
            adv = np.concatenate((x_test_adv, x_test_adv), axis=0)
        else:
            adv = x_test_adv
        n_queries.append(attack.queries)

    predictions = classifier.predict(adv)
    ASR = np.sum(predictions == y_test[:n_images]) / len(y_test[:n_images])
    queries = np.median(n_queries) / np.max(n_queries)
    print(f'ASR: {ASR:.4f}, queries: {queries:.4f}')
    return queries + ASR


def evo_objective(trial):
    global x_test, y_test
    # pop_size = trial.suggest_int('pop_size', 100, 2000)
    # n_gen = trial.suggest_int('n_gen', 10, 200)
    # pixels = trial.suggest_int('pixels', 1, 100)
    alpha = trial.suggest_float('alpha', .001, 1)
    beta = trial.suggest_float('beta', .001, 1)
    gamma = trial.suggest_float('gamma', .001, 1)
    kernel_size = trial.suggest_int('kernel', 1, 5)
    # n_tournament = trial.suggest_int('n_tournament', 2, 50)
    # steps = trial.suggest_int('steps', 50, 1000)

    model = get_model('mobilenet_v2', 'cifar10')

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

    compute_accuracy(classifier)

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
            adv, n_queries, success = EvoAttack(model=model, img=x, label=y, metric='linf', delta=0.4,
                                                perturbed_pixels=32, n_gen=250, pop_size=40,
                                                n_tournament=2, steps=100,
                                                kernel_size=kernel_size, min_pixel=min_pixel_value,
                                                max_pixel=max_pixel_value, alpha=alpha, beta=beta, gamma=gamma).evolve()
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