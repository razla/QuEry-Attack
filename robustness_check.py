import foolbox as fb
import numpy as np
import argparse
import torch

from utils import get_model, correctly_classified, print_initialize, print_success
from data.datasets_loader import load_dataset
from attacks.square_attack import square_attack
from attack import EvoAttack
from utils import compute_accuracy

MODEL_PATH = './models/state_dicts'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn', 'vit_l_16', 'salman', 'gowal', 'rebuffi']
datasets_names = ['mnist', 'imagenet', 'cifar10']

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
    args = parser.parse_args()

    n_images = args.images
    dataset = args.dataset
    model = args.model
    tournament = args.tournament
    eps = args.eps
    pop_size = args.pop
    n_gen = args.gen
    n_iter = n_gen * pop_size

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset)
    init_model = get_model(model, dataset, MODEL_PATH)
    model = fb.PyTorchModel(init_model, bounds=(0, 1))

    count = 0
    success_count = 0
    evo_queries = []
    images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)
        if count == n_images:
            break
        if correctly_classified(dataset, model, x, y) and count < n_images:
            print_initialize(dataset, model, x, y)
            images_indices.append(i)
            adv, n_queries = EvoAttack(dataset=dataset, model=model, x=x, y=y, eps=eps, n_gen=n_gen, pop_size=pop_size,tournament=tournament).generate()
            count += 1
            if not isinstance(adv, type(None)):
                success_count += 1
                adv = adv.cpu().numpy()
                if success_count == 1:
                    evo_x_test_adv = adv
                else:
                    evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
                print_success(dataset, init_model, n_queries, y, adv)
            else:
                print('Evolution failed!')
            evo_queries.append(n_queries)

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
    print('########################################')