import torch.nn as nn
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import inv_normalize_and_save, print_initialize, print_failure, print_success, normalize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EvoAttack():
    def __init__(self, dataset, model, img, label, min_pixel, max_pixel, d_high, d_low, n_gen=1000, pop_size=20, n_tournament=2, steps=100, kernel_size=5, delta=0.3, ):
        self.dataset = dataset
        self.model = model
        self.img = img
        self.shape = img.shape
        self.label = label
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.n_tournament = n_tournament
        self.perturbed_pixels = self.img.shape[2]
        self.steps = steps
        self.kernel_size = kernel_size
        self.delta = delta
        self.fitnesses = torch.zeros(pop_size)
        self.softmax = nn.Softmax(dim=1)
        self.min_ball = torch.tile(torch.maximum(self.img - delta, self.min_pixel), (1, 1))
        self.max_ball = torch.tile(torch.minimum(self.img + delta, self.max_pixel), (1, 1))
        self.current_pop = [self.vertical_mutation(self.img.clone()) for i in range(self.pop_size)]
        self.n_queries = 0
        self.rho = 2
        self.num_plateus = 0
        self.best_individual = None
        self.not_best_individual = None
        self.stop = False
        self.explore = False
        self.low_diversity = d_low
        self.high_diversity = d_high

        print_initialize(self.softmax, self.model, self.img, self.label)

    def get_label_prob(self, individual):
        individual = normalize(self.dataset, individual)
        res = self.softmax(self.model(individual))[0][self.label] - max(x for i, x in enumerate(self.softmax(self.model(individual))[0]) if not i == self.label)
        return res

    def l2_loss(self, individual):
        criterion = nn.MSELoss()
        loss = criterion(individual, self.img)
        return loss

    def linf_loss(self, individual):
        criterion = lambda a, b: torch.norm(a - b, p=float('inf')).item()
        loss = criterion(individual, self.img)
        return loss

    def fitness(self, pop):
        with torch.no_grad():
            self.fitnesses = torch.zeros(self.pop_size)
            for i, individual in enumerate(pop):
                self.n_queries += 1
                if (self.n_queries % self.steps == 0):
                    self.update()
                else:
                    self.fitnesses[i] = self.get_label_prob(individual) + self.l2_loss(individual)
                if (self.check_pred(individual)):
                    self.best_individual = individual.clone()
                    return True
                else:
                    self.not_best_individual = individual.clone()
            return False

    def get_best_individual(self):
        if self.best_individual != None:
            return self.best_individual
        else:
            best_candidate_index = torch.argmin(self.fitnesses)
        return self.current_pop[best_candidate_index]

    def check_pred(self, individual):
        individual = normalize(self.dataset, individual)
        res = self.softmax(self.model(individual)).argmax(dim=1).squeeze() != self.label
        return res

    def stop_criterion(self):
        if (not self.explore and self.fitness(self.current_pop)):
            diversity = self.diversity()
            print(f'Diversity stop criterion: {diversity}')
            self.stop = True
            return True
        return False

    def vertical_mutation(self, individual):
        size = np.asarray(self.shape)
        size[2] = 1
        individual_new = torch.clip(
            individual + self.delta * torch.tensor(np.random.choice([-1, 1], size=size)).cuda(),
            self.min_ball[0],
            self.max_ball[0],
        )
        return individual_new

    def horizontal_mutation(self, individual):
        size = np.asarray(self.shape)
        size[3] = 1
        individual_new = torch.clip(
            individual + self.delta * torch.tensor(np.random.choice([-1, 1], size=size)).cuda(),
            self.min_ball[0],
            self.max_ball[0],
        )
        return individual_new

    def local_mutate(self, individual):
        for pertube in range(self.perturbed_pixels):
            for channel in range(self.shape[1]):
                kernel = self.kernel_size
                i = np.random.randint(0 + kernel, self.shape[2] - kernel)
                j = np.random.randint(0 + kernel, self.shape[3] - kernel)
                local_pertube = torch.tensor(
                    np.random.normal(loc=0, scale = self.rho * self.delta, size=(kernel, kernel))).to(device)
                individual[0][channel][i:i + kernel, j:j + kernel] += local_pertube
        individual[0] = torch.clip(individual[0], self.min_ball[0], self.max_ball[0])
        return individual.to(device)

    def swap(self, individual1, individual2, channel, i, j):
        crossover_idx = i * self.shape[2] + j
        flattened_ind1 = individual1[0][channel].flatten()
        flattened_ind2 = individual2[0][channel].flatten()
        flattened_ind1_prefix = flattened_ind1[: crossover_idx]
        flattened_ind2_prefix = flattened_ind2[: crossover_idx]
        flattened_ind1_suffix = flattened_ind1[crossover_idx:]
        flattened_ind2_suffix = flattened_ind2[crossover_idx:]

        individual1[0][channel] = torch.cat((flattened_ind1_prefix, flattened_ind2_suffix), dim=0).view(self.shape[2],
                                                                                                        self.shape[3])
        individual2[0][channel] = torch.cat((flattened_ind2_prefix, flattened_ind1_suffix), dim=0).view(self.shape[2],
                                                                                                        self.shape[3])
        individual1[0][channel] = torch.clip(individual1[0][channel], self.min_ball[0][channel], self.max_ball[0][channel])
        individual2[0][channel] = torch.clip(individual2[0][channel], self.min_ball[0][channel], self.max_ball[0][channel])

        return individual1, individual2

    def selection(self, random):
        if random:
            random = np.random.randint(0, len(self.current_pop))
            res = self.current_pop[random].clone()
        else:
            tournament = [np.random.randint(0, len(self.current_pop)) for i in range(self.n_tournament)]
            tournament_fitnesses = [self.fitnesses[tournament[i]] for i in range(self.n_tournament)]
            res = self.current_pop[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]].clone()
        return res

    def crossover(self, individual1, individual2):
        for channel in range(self.shape[1]):
            i = np.random.randint(0, self.shape[2])
            j = np.random.randint(0, self.shape[3])
            individual1, individual2 = self.swap(individual1, individual2, channel, i, j)
        return individual1, individual2

    def evolve_new_gen(self):
        new_gen = []
        for i in range(self.pop_size // 2):
            if (self.explore):
                parent1 = self.selection(random=True)
                parent2 = self.selection(random=True)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.vertical_mutation(offspring1)
                offspring2 = self.horizontal_mutation(offspring2)
            else:
                offspring2 = self.selection(random=False)
                offspring1 = self.get_best_individual()
                prob = np.random.rand()
                if (prob > 0.25):
                    offspring2 = self.local_mutate(offspring2)

            new_gen.append(offspring1)
            new_gen.append(offspring2)
        self.current_pop = new_gen

    def diversity(self):
        n = len(self.current_pop)
        average_ind = sum(self.current_pop) / n
        diagonal = ((2 * self.delta)) * np.sqrt(self.shape[1] * self.shape[2] * self.shape[3])
        summation = 0
        for ind in self.current_pop:
            summation += (ind - average_ind) ** 2
        result = torch.sum(torch.sqrt(summation)) / (diagonal * n)
        return result

    def reset(self):
        print('Reset!')
        self.current_pop = [self.vertical_mutation(self.img.clone()) for i in range(self.pop_size)]
        self.n_tournament = 2
        self.perturbed_pixels = self.img.shape[2]
        self.kernel_size = (self.img.cpu().numpy().shape[2] // 2) - 1
        self.num_plateus = 0
        self.rho = 2
        self.explore = False

    def update(self):
        self.perturbed_pixels = max(1, self.perturbed_pixels - 1)
        self.n_tournament = min(self.n_tournament + 1, self.pop_size // 4)
        self.kernel_size = max(3, self.kernel_size - 1)
        self.num_plateus += 1
        self.rho = max(0.15, 2 * ((0.9) ** self.num_plateus))
        print(f'Rho: {self.rho:.5f}')

    def evolve(self):
        gen = 0
        while gen < self.n_gen and not self.stop_criterion():

            if gen % (self.n_gen // 5) == 0:
                self.reset()

            diversity = self.diversity()
            print(f'Diversity: {diversity:.6f}')

            if diversity < self.low_diversity:
                self.explore = True
                print(f'Explore!')
            elif diversity > self.high_diversity:
                self.explore = False
                print(f'Exploit!')

            self.evolve_new_gen()
            gen += 1

        best_individual = self.get_best_individual()
        inv_normalize_and_save(self.dataset, self.img, self.best_individual, self.not_best_individual)
        if not self.stop_criterion():
            print_failure(self.softmax, self.model, self.img, self.n_queries, self.label, best_individual, gen)
        else:
            print_success(self.softmax, self.model, self.img, self.n_queries, self.label, best_individual, gen)
        return best_individual, self.n_queries, self.stop