import torch

from evo_attack import EvoAttack
from train import load_dataset
from densenet import densenet121, densenet161, densenet169
from googlenet import googlenet
from inception import inception_v3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = [densenet121(pretrained=True).to(device),
          densenet161(pretrained=True).to(device),
          densenet169(pretrained=True).to(device),
          googlenet(pretrained=True).to(device),
          inception_v3(pretrained=True).to(device)]

if __name__ == '__main__':
    dataset = 'cifar10'
    # model = torch.load(f'{dataset}_model.pth')
    # Untrained model
    train_loader, test_loader = load_dataset(dataset)
    images, labels = next(iter(train_loader))
    for i in range(len(images)):
        img, label = images[i].unsqueeze(dim=0).to(device), labels[i].to(device)
        for pretrained in models:
            model = pretrained
            model.eval()
            evo_attack = EvoAttack(model, img, label, dataset, metric='l1')
            evolution = evo_attack.evolve()
            evo_attack = EvoAttack(model, img, label, dataset, metric='l2')
            evolution = evo_attack.evolve()
            evo_attack = EvoAttack(model, img, label, dataset, metric='linf')
            evolution = evo_attack.evolve()

