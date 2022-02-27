from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import argparse
import math
import wandb

from evo_attack import EvoAttack
from runner import get_model
from train import load_dataset

import optuna
from optuna.trial import TrialState

def correctly_classified(model, x, y):
    softmax = nn.Softmax(dim=1)
    return y == torch.argmax(softmax(model(x)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def objective(trial):
    pop_size = trial.suggest_int('pop_size', 100, 2000)
    n_gen = trial.suggest_int('n_gen', 10, 200)
    pixels = trial.suggest_int('pixels', 1, 100)
    kernel_size = trial.suggest_int('kernel', 3, 7)
    n_tournament = trial.suggest_int('n_tournament', 2, 50)
    steps = trial.suggest_int('steps', 50, 1000)

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset('cifar10')

    n_images = 100

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
            adv, n_queries, success = EvoAttack(model=model, img=x, label=y, metric='linf', delta=0.4,
                                                perturbed_pixels=pixels, n_gen=n_gen, pop_size=pop_size,
                                                n_tournament=n_tournament, steps=steps,
                                                kernel_size=kernel_size, min_pixel=min_pixel_value,
                                                max_pixel=max_pixel_value).evolve()
            adv = adv.cpu().numpy()
            if i == 0:
                evo_x_test_adv = adv
            else:
                evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
            if success:
                success_count += 1
            queries.append(n_queries)

    x_test, y_test = x_test[correctly_classified_images_indices], y_test[correctly_classified_images_indices]

    ASR = (1 - success_count / len(y_test))
    queries = int(np.median(n_queries) / np.max(n_queries))

    print(f'ASR: {ASR}')
    print(f'Queries: {queries}')
    return ASR - queries

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=500)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
