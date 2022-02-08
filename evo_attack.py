from torchvision.utils import save_image, make_grid
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torch
from piqa import SSIM
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EvoAttack():
    def __init__(self, model, img, label, dataset, targeted_label=None, logits=True, n_gen=1500, pop_size=10,
                 n_tournament=5, verbose=True, perturbed_pixels=10, epsilon=0.03, reg=0.5, metric='SSIM'):
        self.model = model
        self.img = img
        self.label = label
        self.dataset = dataset
        self.targeted_label = targeted_label
        self.logits = logits
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.n_tournament = n_tournament
        self.verbose = verbose
        self.perturbed_pixels = perturbed_pixels
        self.epsilon = epsilon
        self.reg = reg
        self.metric = metric
        self.fitnesses = [1 for i in range(pop_size)]
        self.softmax = nn.Softmax(dim=1)
        self.current_pop = [img.clone() for i in range(pop_size)]
        self.n_queries = 0

        self.best_attacks = []

        if self.verbose:
            print("################################")
            print(f'Correct class: {self.label}')
            print(f'Initial class prediction: {self.model(img).argmax(dim=1).item()}')
            print(f'Initial probability: {self.softmax(self.model(img)).max():.4f}')
            print("################################")

    def get_label_prob(self, individual):
        res = self.softmax(self.model(individual))[0][self.label]
        return res

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

    def ssim_loss(self, individual):
        criterion = SSIM().to(device)
        loss = criterion(individual, self.img)
        return loss

    def compute_fitness(self, pop):
        with torch.no_grad():
            self.fitnesses = []
            for individual in pop:
                self.n_queries += 1
                if self.metric == 'SSIM':
                    if self.logits == True:
                        self.fitnesses.append(self.get_label_prob(individual) - self.reg * self.ssim_loss(individual))
                    else:
                        self.fitnesses.append(
                            (1 - self.check_pred(individual).item()) - self.reg * self.ssim_loss(individual))
                elif self.metric == 'l1':
                    if self.logits == True:
                        self.fitnesses.append(self.get_label_prob(individual) + self.reg * self.l1_loss(individual))
                    else:
                        self.fitnesses.append(
                            (1 - self.check_pred(individual).item()) - self.reg * self.l1_loss(individual))
                elif self.metric == 'l2':
                    if self.logits == True:
                        self.fitnesses.append(self.get_label_prob(individual) + self.reg * self.l2_loss(individual))
                    else:
                        self.fitnesses.append(
                            (1 - self.check_pred(individual).item()) - self.reg * self.l2_loss(individual))
                elif self.metric == 'linf':
                    if self.logits == True:
                        self.fitnesses.append(self.get_label_prob(individual) + self.reg * self.linf_loss(individual))
                    else:
                        self.fitnesses.append(
                            (1 - self.check_pred(individual).item()) - self.reg * self.linf_loss(individual))
                else:
                    raise ValueError('No such loss metric!')
                if (self.check_pred(individual)):
                    return True

            return False

    def get_best_individual(self):
        best_candidate_index = torch.argmin(torch.Tensor(self.fitnesses))
        return self.current_pop[best_candidate_index]

    def check_pred(self, individual):
        return self.softmax(self.model(individual)).argmax(dim=1).squeeze() != self.label

    def stop_criterion(self):
        if (self.compute_fitness(self.current_pop)):
            return True
        return False

    def mutate(self, individual):
        shape = individual.shape
        for pertube in range(self.perturbed_pixels):
            for channel in range(shape[1]):
                i = np.random.randint(0, shape[2])
                j = np.random.randint(0, shape[3])
                current_pixel = individual[0][channel][i][j].cpu()
                # for SSIM loss
                # pixel_pertube = torch.tensor(np.random.uniform(max(0, current_pixel - self.epsilon), min(1, current_pixel + self.epsilon)))
                pixel_pertube = torch.tensor(
                    np.random.uniform(current_pixel - self.epsilon, current_pixel + self.epsilon))
                individual[0][channel][i][j] = pixel_pertube
        return individual.to(device)

    def swap(self, individual1, individual2, channel, i, j):
        shape = individual1.shape
        crossover_idx = i * shape[2] + j
        flattened_ind1 = individual1[0][channel].flatten()
        flattened_ind2 = individual2[0][channel].flatten()
        flattened_ind1_prefix = flattened_ind1[: crossover_idx]
        flattened_ind2_prefix = flattened_ind2[: crossover_idx]
        flattened_ind1_suffix = flattened_ind1[crossover_idx:]
        flattened_ind2_suffix = flattened_ind2[crossover_idx:]
        individual1[0][channel] = torch.cat((flattened_ind1_prefix, flattened_ind2_suffix), dim=0).view(shape[2],
                                                                                                        shape[3])
        individual2[0][channel] = torch.cat((flattened_ind2_prefix, flattened_ind1_suffix), dim=0).view(shape[2],
                                                                                                        shape[3])
        return individual1, individual2

    def selection(self):
        tournament = [np.random.randint(0, len(self.current_pop)) for i in range(self.n_tournament)]
        tournament_fitnesses = [self.fitnesses[tournament[i]] for i in range(self.n_tournament)]
        return self.current_pop[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]].clone()

    def crossover(self, individual1, individual2):
        shape = individual1.shape
        for channel in range(shape[1]):
            i = np.random.randint(0, shape[2])
            j = np.random.randint(0, shape[3])
            individual1, individual2 = self.swap(individual1, individual2, channel, i, j)
        return individual1, individual2

    def evolve_new_gen(self):
        new_gen = []
        for i in range(self.pop_size):
            parent1 = self.selection()
            parent2 = self.selection()
            offspring1, offspring1 = self.crossover(parent1, parent2)
            offspring1 = self.mutate(offspring1)
            new_gen.append(offspring1)
        self.current_pop = new_gen

    def renormalize(self, tensor):
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.2471, 1 / 0.2435, 1 / 0.2616]),
                                       transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                            std=[1., 1., 1.]),
                                       ])
        return invTrans(tensor)

    def evolve(self):
        gen = 0
        while gen < self.n_gen and not self.stop_criterion():
            if gen % 5 == 0:
                self.best_attacks.append(self.get_best_individual()[0])
            self.evolve_new_gen()
            gen += 1
        best_individual = self.get_best_individual()
        if self.best_attacks != []:
            grid_image = make_grid(self.best_attacks)
            figures_path = Path('figures')
            figures_path.mkdir(exist_ok=True)
            save_image(grid_image, figures_path / f'{self.dataset}_grid.png')
            orig_and_best = torch.cat((self.renormalize(self.img), self.renormalize(best_individual)), dim=3)
            save_image(orig_and_best, figures_path / f'test_{self.metric}.png')
        if not self.stop_criterion():
            if self.verbose:
                print("################################")
                print("Evolution failed")
                print(f'Correct class: {self.label}')
                print(f'Current prediction: {self.model(best_individual).argmax(dim=1).item()}')
                print(f'Current probability: {self.softmax(self.model(best_individual))[0][self.label].item():.4f}')
                print("################################")
        else:
            if self.verbose:
                print("################################")
                print(f'Evolution succeeded in gen #{gen + 1}')
                print(f'Correct class: {self.label}')
                print(f'Current prediction: {self.model(best_individual).argmax(dim=1).item()}')
                print(f'Current probability: {self.softmax(self.model(best_individual))[0][self.label].item():.4f}')
                l_infinity = torch.norm(self.img - best_individual, p=float('inf')).item()
                print(f'L infinity: {l_infinity:.4f}')
                print(f'Number of queries: {self.n_queries}')
                print("################################")
