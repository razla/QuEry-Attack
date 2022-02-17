import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import argparse
import math

from art.attacks.evasion import FastGradientMethod, ZooAttack, UniversalPerturbation, FeatureAdversariesPyTorch, DeepFool
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset
from evo_attack import EvoAttack
from model import ConvNet
from runner import get_model

dataset = 'cifar10'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models_names = ['custom', 'densenet121', 'densenet161', 'densenet169', 'googlenet', 'inception_v3']
datasets_names = ['cifar10', 'mnist']
metrics = ['l2', 'linf']


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 1: Load the MNIST dataset

def load_data(ds_name):
    return load_dataset(ds_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=['ALL'] + models_names, default='custom',
                        help="Run only specific model, or 'ALL' models")
    parser.add_argument("--dataset", "-d", choices=['ALL'] + datasets_names, default='cifar10',
                        help="Run only specific dataset, or 'ALL' datasets")
    parser.add_argument("--metric", "-t", choices=['ALL'] + metrics, default='ALL',
                        help="Use only specific metric; or 'ALL' metrics")
    parser.add_argument("--images", "-i", type=int, default=None,
                        help="Maximal number of images from dataset to process (or None, to process all)")
    parser.add_argument("--local", "-l", action='store_true',
                        help='Runs on local machine (default: use Slurm if available)')
    args = parser.parse_args()

    if args.model == 'ALL':
        models = models_names
    else:
        models = [args.model]

    if args.dataset =='ALL':
        datasets = datasets_names
    else:
        dataset = [args.dataset]

    if args.metric != 'ALL':
        metrics = [args.metric]

    if args.images is None:
        args.images = math.inf

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_data(dataset)

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    x_test = x_test[:20]
    y_test = y_test[:20]

    model = get_model('custom', 'cifar10')

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

    # Step 4: Train the ART classifier

    # classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)

    # Craft adversarial samples with DeepFool
    print("Create DeepFool attack")
    adv_crafter = DeepFool(classifier)
    print("Craft attack on training examples")
    x_train_adv = adv_crafter.generate(x_train[:250])
    print("Craft attack test examples")
    x_test_adv = adv_crafter.generate(x_test)

    # Evaluate the classifier on the adversarial samples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Classifier before adversarial training")
    print("Accuracy on adversarial samples: %.2f%%", (acc * 100))

    # Data augmentation: expand the training set with the adversarial samples
    x_train = np.append(x_train, x_train_adv, axis=0)
    y_train = np.append(y_train, y_train, axis=0)

    # Retrain the CNN on the extended dataset
    # classifier.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    # zoo_attack = ZooAttack(classifier=classifier)
    # ftr_adv_attack = FeatureAdversariesPyTorch(estimator=classifier, step_size=0.1, delta=0.2)
    univ_attack = UniversalPerturbation(classifier=classifier, norm='inf', eps=0.2)
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = torch.from_numpy(x).unsqueeze(dim=0).to(device)
        y = torch.tensor(np.argmax(y)).to(device)
        adv = EvoAttack(model=model, img=x, label=y, metric='linf', delta=0.2, perturbed_pixels=1,
                        kernel_size=3).evolve().cpu().numpy()
        if i == 0:
            evo_x_test_adv = adv
        else:
            evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
    # zoo_x_test_adv = zoo_attack.generate(x=x_test)
    # ftr_adv_x_test_adv = ftr_adv_attack.generate(x=x_test, y=y_test)
    univ_x_test_adv = univ_attack.generate(x=x_test, y=y_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    # predictions = classifier.predict(zoo_x_test_adv)
    # predictions = classifier.predict(ftr_adv_x_test_adv)
    predictions = classifier.predict(univ_x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on Universal Perturbation adversarial test examples: {}%".format(accuracy * 100))
    predictions = classifier.predict(evo_x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on Evo adversarial test examples: {}%".format(accuracy * 100))