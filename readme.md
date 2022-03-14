
## Prepare the environment
    conda create -n evo_attack numpy pytorch
    conda activate evo_attack
    pip install piqa

## Download the trained models
- Download model from Google Drive link at https://github.com/huyvnphan/PyTorch_CIFAR10
- Unzip it, to get `state_dicts/*.pt`

## Training custom models
- Train models for required datasets (currently only `cifar10`), and put them in `models/state_dicts`
- python training_field.py --model=custom --dataset=mnist --epochs=100 --lr=0.01 --weight_dec=1e-6

## Run
- python testing_field.py --model=custom --dataset=mnist --delta=0.05 --image=200 --pop=40 --gen=500 --d_low=1e-6 --d_high=2e-1
