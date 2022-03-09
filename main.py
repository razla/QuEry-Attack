import argparse
import math
import subprocess
import sys
import os
import torch

from datasets_loader import load_dataset

dataset = 'cifar10'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models_names = ['custom', 'densenet121', 'densenet161', 'densenet169', 'googlenet', 'inception_v3']
metrics = ['l0', 'l2', 'linf', 'SSIM']


def is_slurm_available():
    try:
        return subprocess.run(['which', 'sbatch'], capture_output=True).returncode == 0
    except:
        return False


def run_attack(local, model_name, i, dataset, metric):
    command = f'{sys.executable} runner.py {model_name} {i} {dataset} {metric}'
    if local or not is_slurm_available():
        subprocess.run(command.split())
    else:
        # TODO monitor the job? report when finished?
        if os.environ['HOSTNAME'] == 'cpu-s-master':
            subprocess.run(['sbatch', f'--wrap={command}'])
        else:
            subprocess.run(['sbatch', '--gpus=1', f'--wrap={command}'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--model", "-m", choices=['ALL'] + models_names, default='custom',
                        help="Run only specific model, or 'ALL' models")
    parser.add_argument("--metric", "-t", choices=['ALL'] + metrics, default='ALL',
                        help="Use only specific metric; or 'ALL' metrics")
    parser.add_argument("--images", "-i", type=int, default=None,
                        help="maximal number of images from dataset to process (or None, to process all)")
    parser.add_argument("--local", "-l", action='store_true',
                        help='runs on local machine (default: use Slurm if available)')
    args = parser.parse_args()

    if args.model == 'ALL':
        models = models_names
    else:
        models = [args.model]

    if args.metric != 'ALL':
        metrics = [args.metric]

    if args.images is None:
        args.images = math.inf

    train_loader, _ = load_dataset(dataset)
    images, labels = next(iter(train_loader))
    for i in range(min(len(images), args.images)):
        for model in models:
            for metric in metrics:
                run_attack(args.local, model, i, dataset, metric=metric)
