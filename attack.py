from torchvision.utils import save_image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path
import random
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EvoAttack():
    def __init__(self, model, img, label, min_pixel, max_pixel, targeted_label=None, logits=True, n_gen=1000, pop_size=20,
                 n_tournament=2, verbose=True, perturbed_pixels=1, epsilon=1, alpha=1, beta=1, gamma=1, metric='linf',
                 epsilon_decay=0.99, steps=100, kernel_size=5, delta=0.3):
        self.model = model
        self.img = img
        self.label = label
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.targeted_label = targeted_label
        self.logits = logits
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.n_tournament = n_tournament
        self.verbose = verbose
        self.perturbed_pixels = perturbed_pixels
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.metric = metric
        self.epsilon_decay = epsilon_decay
        self.steps = steps
        self.kernel_size = kernel_size
        self.delta = delta
        self.fitnesses = torch.zeros(pop_size)
        self.softmax = nn.Softmax(dim=1)
        self.min_ball = torch.tile(torch.maximum(self.img - delta, self.min_pixel), (1, 1))
        self.max_ball = torch.tile(torch.minimum(self.img + delta, self.max_pixel), (1, 1))
        self.current_pop = [self.local_mutate(self.img.clone()) for i in range(pop_size)]
        self.n_queries = 0
        self.best_individual = None
        self.best_attacks = []
        self.stop = False
        self.explore = False

        if self.verbose:
            print("################################")
            print(f'Correct class: {self.label}')
            print(f'Targeted class: {self.targeted_label}')
            print(f'Initial class prediction: {self.model(self.img).argmax(dim=1).item()}')
            print(f'Initial probability: {self.softmax(self.model(img)).max():.4f}')
            print("################################")

    def get_label_prob(self, individual):
        res = self.softmax(self.model(individual))[0][self.label] - sum(x for i, x in enumerate(self.softmax(self.model(individual))[0]) if not i == self.label)
        return res

    def get_targeted_label_prob(self, individual):
        res = self.softmax(self.model(individual))[0][self.targeted_label] - sum(x for i, x in enumerate(self.softmax(self.model(individual))[0]) if not i == self.targeted_label)
        return res

    def l0_loss(self, individual):
        flattened_ind = individual.flatten()
        flattened_img = self.img.flatten()
        length = len(flattened_ind)
        loss = length - sum(torch.isclose(flattened_ind, flattened_img))
        return loss

    def l1_loss(self, individual):
        criterion = nn.L1Loss()
        loss = criterion(individual, self.img)
        return loss

    def l2_loss(self, individual):
        criterion = nn.MSELoss()
        loss = criterion(individual, self.img)
        return loss

    def linf_loss(self, individual):
        criterion = lambda a, b: torch.norm(a - b, p=float('inf')).item()
        loss = criterion(individual, self.img)
        return loss

    def compute_fitness(self, pop):
        with torch.no_grad():
            self.fitnesses = torch.zeros(self.pop_size)
            for i, individual in enumerate(pop):
                self.n_queries += 1
                if (self.n_queries % self.steps == 0):
                    self.perturbed_pixels = max(1, self.perturbed_pixels - 1)
                    self.n_tournament += min(self.n_tournament + 1, self.pop_size // 2)
                    # self.epsilon *= self.epsilon
                elif self.metric == 'l0':
                    if self.targeted_label == None:
                        self.fitnesses[i] = self.alpha * self.get_label_prob(individual) + self.beta * self.l0_loss(individual) + self.gamma * self.ind_diversity(individual)
                    else:
                        self.fitnesses[i] = self.alpha * self.get_targeted_label_prob(individual) - self.beta * self.l1_loss(individual) - self.gamma * self.ind_diversity(individual)
                elif self.metric == 'l2':
                    if self.targeted_label == None:
                        self.fitnesses[i] = self.alpha * self.get_label_prob(individual) + self.beta * self.l2_loss(individual) + self.gamma * self.ind_diversity(individual)
                    else:
                        self.fitnesses[i] = self.alpha * self.get_targeted_label_prob(individual) - self.beta * self.l2_loss(individual) - self.gamma * self.ind_diversity(individual)
                elif self.metric == 'linf':
                    if self.targeted_label == None:
                        self.fitnesses[i] = self.alpha * self.get_label_prob(individual) + self.beta * self.l2_loss(individual) + self.gamma * self.ind_diversity(individual)
                    else:
                        self.fitnesses[i] = self.alpha * self.get_targeted_label_prob(individual) - self.beta * self.l2_loss(individual) + self.gamma * self.ind_diversity(individual)
                else:
                    raise ValueError('No such loss metric!')
                if (self.check_pred(individual)):
                    self.best_individual = individual.clone()
                    return True

            return False

    def get_best_individual(self):
        if self.best_individual != None:
            return self.best_individual
        elif self.targeted_label == None:
            best_candidate_index = torch.argmin(self.fitnesses)
        else:
            best_candidate_index = torch.argmax(self.fitnesses)
        return self.current_pop[best_candidate_index]

    def check_pred(self, individual):
        if self.targeted_label == None:
            res = self.softmax(self.model(individual)).argmax(dim=1).squeeze() != self.label
        else:
            res = self.softmax(self.model(individual)).argmax(dim=1).squeeze() == self.targeted_label
        return res

    def stop_criterion(self):
        if (self.compute_fitness(self.current_pop)):
            diversity = self.compute_whole_diversity()
            print(f'Diversity stop criterion: {diversity}')
            self.stop = True
            return True
        return False

    def local_mutate(self, individual):
        shape = individual.shape
        for pertube in range(self.perturbed_pixels):
            for channel in range(shape[1]):
                i = np.random.randint(0 + self.kernel_size, shape[2] - self.kernel_size)
                j = np.random.randint(0 + self.kernel_size, shape[3] - self.kernel_size)
                local_pertube = torch.tensor(np.random.uniform(- self.delta, self.delta, (self.kernel_size, self.kernel_size))).to(device)
                individual[0][channel][i:i + self.kernel_size, j:j + self.kernel_size] += local_pertube
                individual[0][channel] = torch.clip(individual[0][channel], self.min_ball[0][channel], self.max_ball[0][channel])
        return individual.to(device)

    def mutate(self, individual):
        shape = individual.shape
        for pertube in range(self.perturbed_pixels):
            for channel in range(shape[1]):
                i = np.random.randint(0, shape[2])
                j = np.random.randint(0, shape[3])
                current_pixel = individual[0][channel][i][j].cpu()
                pixel_pertube = torch.tensor(
                    np.random.uniform(current_pixel - self.epsilon, current_pixel + self.epsilon))
                individual[0][channel][i][j] = pixel_pertube
        return individual.to(device)

    def swap(self, individual1, individual2, channel, i, j):
        shape = individual1.shape
        crossover_idx = i * shape[2] + j
        flattened_ind1 = individual1[0][channel].flatten()
        flattened_ind2 = individual2[0][channel].flatten()
        flattened_ind1_prefix =  flattened_ind1[: crossover_idx] # * self.epsilon
        flattened_ind2_prefix = flattened_ind2[: crossover_idx] # * self.epsilon
        flattened_ind1_suffix = flattened_ind1[crossover_idx:] # * self.epsilon
        flattened_ind2_suffix = flattened_ind2[crossover_idx:] # * self.epsilon
        individual1[0][channel] = torch.cat((flattened_ind1_prefix, flattened_ind2_suffix), dim=0).view(shape[2],
                                                                                                        shape[3])
        individual2[0][channel] = torch.cat((flattened_ind2_prefix, flattened_ind1_suffix), dim=0).view(shape[2],
                                                                                                        shape[3])
        individual1[0][channel] = torch.clip(individual1[0][channel], self.min_ball[0][channel], self.max_ball[0][channel])

        individual2[0][channel] = torch.clip(individual2[0][channel], self.min_ball[0][channel], self.max_ball[0][channel])

        return individual1, individual2

    def selection(self):
        tournament = [np.random.randint(0, len(self.current_pop)) for i in range(self.n_tournament)]
        tournament_fitnesses = [self.fitnesses[tournament[i]] for i in range(self.n_tournament)]
        if self.targeted_label == None:
            res = self.current_pop[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]].clone()
        else:
            res = self.current_pop[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]].clone()
        return res

    def crossover(self, individual1, individual2):
        shape = individual1.shape
        for channel in range(shape[1]):
            i = np.random.randint(0, shape[2])
            j = np.random.randint(0, shape[3])
            individual1, individual2 = self.swap(individual1, individual2, channel, i, j)
        return individual1, individual2

    def evolve_new_gen(self):
        new_gen = []
        for i in range(self.pop_size // 2):
            parent1 = self.selection()
            parent2 = self.selection()
            if (self.explore):
                parent1, parent2 = self.crossover(parent1, parent2)
            # if (self.explore):
            elitist = self.get_best_individual()
            offspring1 = self.local_mutate(parent1)
            # offspring2 = self.local_mutate(parent2)
            new_gen.append(elitist)
            new_gen.append(offspring1)
        self.current_pop = new_gen

    def renormalize(self, tensor):
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.2471, 1 / 0.2435, 1 / 0.2616]),
                                       transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                            std=[1., 1., 1.]),
                                       ])
        return invTrans(tensor)

    def ind_diversity(self, individual):
        diversity = []
        for other in self.current_pop:
            if torch.sum(individual != other) != 0:
                diversity.append(torch.sum((individual - other) ** 2))
        diversity = max(diversity) / sum(diversity)
        return diversity

    def compute_whole_diversity(self):
        diversity = []
        for ind in self.current_pop:
            diversity.append(self.ind_diversity(ind))
        diversity = max(diversity) / sum(diversity)
        return diversity

    def evolve(self):
        gen = 0
        i = 0
        best_diversity = -math.inf
        while gen < self.n_gen and not self.stop_criterion():
            diversity = self.compute_whole_diversity()

            if diversity > best_diversity:
                best_diversity = diversity
                self.explore = False
                print(f'Exploit!')
            else:
                i += 1

            if i == 10:
                i = 0
                self.explore = True
                print(f'Best diversity: {best_diversity}')
                # best_diversity = diversity
                print(f'Explore!')

            if gen % 5 == 0:
                print(f'Diversity gen #{gen}: {diversity}')
                self.best_attacks.append(self.get_best_individual()[0])
            self.evolve_new_gen()
            gen += 1
        best_individual = self.get_best_individual()
        if self.best_attacks != []:
            figures_path = Path('figures')
            figures_path.mkdir(exist_ok=True)
            orig_and_best = torch.cat((self.img, best_individual), dim=3)
            save_image(orig_and_best, figures_path / f'test_{self.metric}_{random.randrange(0,400)}.png')
        if not self.stop_criterion():
            if self.verbose:
                print("################################")
                print("Evolution failed")
                print(f'Correct class: {self.label}')
                print(f'Current prediction: {self.model(best_individual).argmax(dim=1).item()}')
                print(f'Current probability (orig class): {self.softmax(self.model(best_individual))[0][self.label].item():.4f}')
                if self.targeted_label:
                    print(
                        f'Current probability (target class): {self.softmax(self.model(best_individual))[0][self.targeted_label].item():.4f}')
                print("################################")
        else:
            if self.verbose:
                print("################################")
                print(f'Evolution succeeded in gen #{gen + 1}')
                print(f'Correct class: {self.label}')
                print(f'Current prediction: {self.model(best_individual).argmax(dim=1).item()}')
                print(f'Current probability (orig class): {self.softmax(self.model(best_individual))[0][self.label].item():.4f}')
                if self.targeted_label:
                    print(
                        f'Current probability (target class): {self.softmax(self.model(best_individual))[0][self.targeted_label].item():.4f}')
                l_infinity = torch.norm(self.img - best_individual, p=float('inf')).item()
                print(f'L infinity: {l_infinity:.4f}')
                print(f'Number of queries: {self.n_queries}')
                print("################################")
        return best_individual, self.n_queries, self.stop