# QuEry Attack

## This paper was accepted to a special issue Algorithms for Natural Computing Models - Algorithms journal

Code accompanying the paper:
[An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Convolutional Neural Networks", Algorithms journal, 2022](https://www.mdpi.com/1999-4893/15/11/407).

### Abstract:
Deep neural networks (DNNs) are sensitive
to adversarial data in a variety of scenarios, including
the black-box scenario, where the attacker is only al-
lowed to query the trained model and receive an output.
Existing black-box methods for creating adversarial
instances are costly, often using gradient estimation or
training a replacement network. This paper introduces
Query-Efficient Evolutionary Attack, QuEry Attack, a
score-based, black-box attack. QuEry Attack is based
on a novel objective function that can be used in
gradient-free optimization problems. The attack only
requires access to the output logits of the classifier and
is thus not affected by gradient masking. No additional
information is needed, rendering our method more
suitable to real-life situations. We test its performance
with three different state-of-the-art models—Inception-
v3, ResNet-50, and VGG-16-BN—against three bench-
mark datasets: MNIST, CIFAR10 and ImageNet. Fur-
thermore, we evaluate QuEry Attack’s performance
on non-differential transformation defenses and state-
of-the-art robust models. Our results demonstrate the
superior performance of QuEry Attack, both in terms
of accuracy score and query efficiency.

## Prerequisites
    conda create -n query_attack python=3.8.12
    pip install -r requirements.txt

## Download the trained models weights
2. Download models' weights from this link: [models weights](https://drive.google.com/file/d/1LKLicAXgL-Q9QFtvMWDkHN-8ESPBNjtO/view?usp=sharing)
3. Unzip it and place it in models/state_dicts/*.pt

## Run
    python main.py --model=<model_name> --dataset=<dataset_name> --eps=<epsilon> --pop=<pop_size> --gen=<n_gen> --images=<n_images> --tournament=<n_tournament> --path=<imagenet_path>
- For MNIST dataset, run the above command with --model=custom
- The values used in the paper are:
  - pop_size=70
  - gen=600
  - images=200
  - tournament=25

If you wish to cite this paper:
```
@Article{a15110407,
AUTHOR = {Lapid, Raz and Haramaty, Zvika and Sipper, Moshe},
TITLE = {An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Convolutional Neural Networks},
JOURNAL = {Algorithms},
VOLUME = {15},
YEAR = {2022},
NUMBER = {11},
ARTICLE-NUMBER = {407},
URL = {https://www.mdpi.com/1999-4893/15/11/407},
ISSN = {1999-4893},
ABSTRACT = {Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces Qu ery-Efficient Evolutionary Attack&mdash;QuEry Attack&mdash;an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different, commonly used, pretrained image-classifications models&mdash;Inception-v3, ResNet-50, and VGG-16-BN&mdash;against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack&rsquo;s performance on non-differential transformation defenses and robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.},
DOI = {10.3390/a15110407}
}
```
![alt text](https://github.com/razla/QuEry-Attack/blob/master/figures/examples.png)
