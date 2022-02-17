"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch

from art.attacks.evasion import FastGradientMethod, ZooAttack, UniversalPerturbation, FeatureAdversariesPyTorch, DeepFool
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10
from evo_attack import EvoAttack
from model import ConvNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 0: Define the neural network model, return logits instead of activation in forward method

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
#         self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
#         self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
#         self.fc_2 = nn.Linear(in_features=100, out_features=10)
#
#     def forward(self, x):
#         x = F.relu(self.conv_1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 10)
#         x = F.relu(self.fc_1(x))
#         x = self.fc_2(x)
#         return x


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

x_test = x_test[:20]
y_test = y_test[:20]

# Step 2: Create the model

model = ConvNet(in_channels=3).to(device)

# Step 2a: Define the loss function and the optimizer

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

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)

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
    adv = EvoAttack(model=model, img=x, label=y, metric='linf', delta=0.2, perturbed_pixels=1, kernel_size=3).evolve().cpu().numpy()
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