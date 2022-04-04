from torchvision.utils import save_image
import torch.nn.functional as F
from operator import itemgetter
import numpy as np
import random
import torch
from copy import deepcopy

from utils import normalize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EvoAttack():
    def __init__(self, dataset, model, x, y, n_gen=500, pop_size=40, eps=0.3, tournament=35, defense=False):
        self.dataset = dataset
        self.model = model
        self.x = x
        self.y = y
        self.p_init = 0.1
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.n_tournament = tournament
        self.eps = eps
        self.best_x_hat = None
        self.queries = 0
        self.defense = defense
        self.min_ball = torch.tile(torch.maximum(self.x - eps, torch.tensor(0)), (1, 1))
        self.max_ball = torch.tile(torch.minimum(self.x + eps, torch.tensor(1)), (1, 1))

    def generate(self):
        save_image(self.x, 'orig.jpg')
        gen = 0
        cur_pop = self.init()
        while not self.termination_condition(cur_pop, gen):
            self.fitness(cur_pop)
            new_pop = []
            elite = self.elitism(cur_pop)

            new_pop.append([elite, np.inf])
            for i in range((self.pop_size - 1) // 3):
                parent1 = self.selection(cur_pop)
                parent2 = self.selection(cur_pop)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                mut_offspring1 = self.mutation((offspring1, np.inf))
                mut_offspring2 = self.mutation((offspring2, np.inf))
                offspring1 = self.project(offspring1)
                new_pop.append([offspring1, np.inf])
                new_pop.append([mut_offspring1, np.inf])
                new_pop.append([mut_offspring2, np.inf])

            print(f'Elitist: {min(cur_pop, key=itemgetter(1))[1]:.5f}')

            cur_pop = new_pop
            gen += 1

        return self.best_x_hat, self.queries

    def crossover(self, parent1, parent2):
        parent1 = parent1[0].flatten()
        parent2 = parent2[0].flatten()
        i = np.random.randint(0, len(parent1))
        j = np.random.randint(i, len(parent1))
        offspring1 = torch.cat([parent1[:i], parent2[i:j], parent1[j:]], dim = 0)
        offspring2 = torch.cat([parent2[:i], parent1[i:j], parent2[j:]], dim = 0)
        offspring1 = offspring1.reshape(self.x.shape)
        offspring2 = offspring2.reshape(self.x.shape)
        offspring1 = self.project(offspring1)
        offspring2 = self.project(offspring2)
        return offspring1, offspring2


    def elitism(self, cur_pop):
        elite = min(cur_pop, key=itemgetter(1))[0]
        return elite

    def selection(self, cur_pop):
        selection = [random.choice(cur_pop) for i in range(self.n_tournament)]
        best = min(selection, key=itemgetter(1))
        return best

    def mutation(self, x_hat):
        p = self.p_selection(self.p_init, self.queries, self.n_gen * self.pop_size)
        c = x_hat[0].shape[1]
        h = x_hat[0].shape[2]
        w = x_hat[0].shape[3]
        n_features = c * h * w
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        x_curr_window = x_hat[0][:, :, center_h:center_h + s, center_w:center_w + s]
        for i in range(c):
            x_curr_window[:, i] += np.random.choice([-2 * self.eps, 2 * self.eps]) * torch.ones(x_curr_window[:, i].shape).to(device)

        x_hat[0][:, :, center_h:center_h + s, center_w: center_w + s] = x_curr_window
        x_hat = self.project(x_hat[0])
        return x_hat

    def fitness(self, cur_pop):
        for i in range(len(cur_pop)):
            x_hat, fitness = cur_pop[i]
            x_hat_l2 = F.mse_loss(x_hat, self.x)
            def_x_hat = x_hat.clone()
            if self.defense:
                def_x_hat = self.defense(def_x_hat.cpu().numpy())[0]
                def_x_hat = torch.tensor(def_x_hat).to(device)
            n_x_hat = normalize(self.dataset, def_x_hat)
            probs = F.softmax(self.model(n_x_hat), dim = 1).squeeze()
            objective = probs[self.y] - max(x for i, x, in enumerate(probs) if not i == self.y)
            cur_pop[i] = [x_hat, objective + x_hat_l2]

    def get_label(self, x_hat):
        def_x_hat = x_hat.clone()
        if self.defense:
            def_x_hat = self.defense(def_x_hat.cpu().numpy())[0]
            def_x_hat = torch.tensor(def_x_hat).to(device)
        n_x_hat = normalize(self.dataset, def_x_hat)
        return torch.argmax(F.softmax(self.model(n_x_hat), dim = 1))

    def termination_condition(self, cur_pop, gen):

        if gen == self.n_gen:
            return True
        for [x_hat, _] in cur_pop:
            y_hat = self.get_label(x_hat)
            self.queries += 1
            if y_hat != self.y:
                self.best_x_hat = x_hat
                save_image(x_hat, 'good.jpg')
                return True
            else:
                save_image(x_hat, 'bad.jpg')
        return False

    def project(self, x_hat):
        projected_x_hat = torch.clip(x_hat, self.min_ball, self.max_ball)
        return projected_x_hat

    def init(self):
        cur_pop = []
        for i in range(self.pop_size):
            x_hat = self.x.clone()
            x_hat = self.vertical_mutation(x_hat)

            cur_pop.append([x_hat, np.inf])
        return cur_pop

    def vertical_mutation(self, x_hat):
        size = np.asarray(self.x.shape)
        size[2] = 1
        x_hat = x_hat + self.eps * torch.tensor(np.random.choice([-1, 1], size=size)).to(device)
        x_hat = self.project(x_hat)
        return x_hat

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p