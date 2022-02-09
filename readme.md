
## Prepare the environment
    conda create -n evo_attack numpy pytorch
    conda activate evo_attack
    pip install piqa

## Download the trained models
- Download model from Google Drive link at https://github.com/huyvnphan/PyTorch_CIFAR10
- Unzip it, to get `state_dicts/*.pt`

## Training custom models
- Train models for required datasets (currently only `cifar10`), and put them in `models/`
- `TODO`: how exactly?

## Run
`python main.py`
