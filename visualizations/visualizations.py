from torchvision.utils import save_image
import numpy as np
import argparse
import torch
import sys

sys.path.append('/cs_storage/razla/Attack/evo_attack/')

print(sys.path)

from attack import EvoAttack
from utils import get_model, correctly_classified, print_initialize
from data.datasets_loader import load_dataset

from time import time

MODEL_PATH = '../../models/state_dicts'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn']
datasets_names = ['mnist', 'imagenet', 'cifar10', 'svhn']
metrics = ['l2', 'linf']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=models_names, default='custom',
                        help="Run only specific model")
    parser.add_argument("--dataset", "-da", choices=datasets_names, default='cifar10',
                        help="Run only specific dataset")
    parser.add_argument("--eps", "-ep", type=float, default=0.1,
                        help="Constrained optimization problem - epsilon")
    parser.add_argument("--pop", "-pop", type=int, default=20,
                        help="Population size")
    parser.add_argument("--gen", "-g", type=int, default=1000,
                        help="Number of generations")
    parser.add_argument("--images", "-i", type=int, default=3,
                        help="Maximal number of images from dataset to process")
    parser.add_argument("--tournament", "-t", type=int, default=35,
                        help="Tournament selection")
    parser.add_argument("--target", "-ta", type=int, default=1,
                        help="Target label")
    args = parser.parse_args()

    n_images = args.images
    dataset = args.dataset
    model = args.model
    tournament = args.tournament
    eps = args.eps
    pop_size = args.pop
    n_gen = args.gen
    targeted = args.target
    n_iter = n_gen * pop_size

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset)
    init_model = get_model(model, dataset, MODEL_PATH)

    count = 0
    success_count = 0
    evo_queries = []
    evo_times = []
    images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if count == n_images:
            break

        if y == torch.tensor(targeted) and correctly_classified(dataset, init_model, x, y) and count < n_images:
            print_initialize(dataset, init_model, x, y)
            targets = [i for i in range(10) if i != y.item()]
            # targets = [9]
            num_success = 0
            for target in targets:
                save_image(x, f'{y}_{y}.png')
                images_indices.append(i)
                start = time()
                adv, n_queries = EvoAttack(dataset=dataset, model=init_model, x=x, y=y, eps=eps, n_gen=n_gen, pop_size=pop_size,tournament=tournament, targeted=target).generate()
                end = time()
                overall = end - start

                if not isinstance(adv, type(None)):
                    save_image(adv, f'{y}_{target}.png')
                    adv = adv.cpu().numpy()
                    num_success += 1
                    success_count += 1
                    print('Success!')
                evo_times.append(overall)
                evo_queries.append(n_queries)

            if num_success == 9:
                count += 1
    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {dataset}')
    print(f'\tModel: {model}')
    print(f'\tTournament: {tournament}')
    print(f'\tMetric: linf, epsilon: {eps:.4f}')
    print(f'\tEvo:')
    print(f'\t\tEvo - test accuracy: {(1 - (success_count / n_images)) * 100:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    print(f'\t\tEvo - times (median): {np.median(evo_times)}')
    print('########################################')