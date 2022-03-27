from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SquareAttack
import torch.nn as nn
import numpy as np
# import torchattacks

from utils import get_normalization

# def pgd_attack(dataset, model, x_test, y_test, i, square_adv, n_iter, delta):
#     atk = torchattacks.PGD(model, eps=delta, alpha=2 / 255, steps=4)
#     adv_images = atk(x_test, y_test.unsqueeze(dim=0))
#     return adv_images, atk.queries

def square_attack(dataset, init_model, min_ball, max_ball, x_test, i, square_adv, n_iter, delta):
    square_classifier = PyTorchClassifier(
        model=init_model,
        clip_values=(min_ball.numpy(), max_ball.numpy()),
        loss=nn.CrossEntropyLoss(),
        input_shape=x_test[0].numpy().shape,
        nb_classes=10,
        preprocessing=get_normalization(dataset)
    )

    square_attack = SquareAttack(estimator=square_classifier, norm='inf', max_iter=n_iter, p_init=0.05, eps=delta)
    square_x_test_adv = square_attack.generate(x=x_test.numpy())
    if i != 0:
        square_adv = np.concatenate((square_x_test_adv, square_adv), axis=0)
    else:
        square_adv = square_x_test_adv
    return square_adv, square_attack.queries