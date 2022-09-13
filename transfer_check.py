import torch.nn.functional as F
import numpy as np
import argparse
import torch

from utils import get_model, correctly_classified, print_initialize, print_success, normalize
from data.datasets_loader import load_dataset
from attack import EvoAttack

MODEL_PATH = './models/state_dicts'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn', 'vit_l_16']
datasets_names = ['mnist', 'imagenet', 'cifar10']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--s_model", "-sm", choices=models_names, default='inception_v3',
                        help="Source model")
    parser.add_argument("--t_model", "-tm", choices=models_names, default='resnet50',
                        help="Target model")
    parser.add_argument("--dataset", "-da", choices=datasets_names, default='imagenet',
                        help="Run only specific dataset")
    parser.add_argument("--eps", "-ep", type=float, default=0.075,
                        help="Constrained optimization problem - epsilon")
    parser.add_argument("--pop", "-pop", type=int, default=20,
                        help="Population size")
    parser.add_argument("--gen", "-g", type=int, default=1000,
                        help="Number of generations")
    parser.add_argument("--images", "-i", type=int, default=20,
                        help="Maximal number of images from dataset to process")
    parser.add_argument("--tournament", "-t", type=int, default=35,
                        help="Tournament selection")
    parser.add_argument("--path", "-ip", default='/cs_storage/public_datasets/ImageNet',
                        help="ImageNet dataset path")
    args = parser.parse_args()

    n_images = args.images
    dataset = args.dataset
    sm = args.s_model
    tm = args.t_model

    tournament = args.tournament
    eps = args.eps
    pop_size = args.pop
    n_gen = args.gen
    imagenet_path = args.path
    n_iter = n_gen * pop_size

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset, imagenet_path)
    s_model = get_model(sm, dataset, MODEL_PATH)
    t_model = get_model(tm, dataset, MODEL_PATH)
    count = 0
    success_count = 0
    transfer_success_count = 0
    evo_queries = []
    images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if count == n_images:
            break

        if correctly_classified(dataset, s_model, x, y) and correctly_classified(dataset, t_model, x, y) and count < n_images:
            print_initialize(dataset, s_model, x, y)
            print_initialize(dataset, t_model, x, y)
            count += 1
            images_indices.append(i)
            adv, n_queries = EvoAttack(dataset=dataset, model=s_model, x=x, y=y, eps=eps, n_gen=n_gen, pop_size=pop_size,tournament=tournament).generate()

            if not isinstance(adv, type(None)):
                success_count += 1

                n_adv = normalize(dataset, adv)
                pred = torch.argmax(F.softmax(t_model(n_adv), dim=1).squeeze())
                if pred != y:
                    transfer_success_count += 1
                    print('Transferred successfully!')
                else:
                    print('Transferred unsuccessfully!')

                adv = adv.cpu().numpy()
                if success_count == 1:
                    evo_x_test_adv = adv
                else:
                    evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
                print_success(dataset, s_model, n_queries, y, adv)

            else:
                print('Evolution failed!')
            evo_queries.append(n_queries)


    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {dataset}')
    print(f'\tSource model: {sm}')
    print(f'\tTarget model: {tm}')
    print(f'\tTournament: {tournament}')
    print(f'\tMetric: linf, epsilon: {eps:.4f}')
    print(f'\tEvo:')
    print(f'\t\tEvo - test accuracy: {(1 - (success_count / n_images)) * 100:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    print(f'\t\tTransfer success count: {transfer_success_count}')
    print('########################################')
