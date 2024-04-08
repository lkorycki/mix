# MIX

Source code for: *Class-Incremental Mixture of Gaussians for Deep Continual Learning*.

## Setup

Python version: 3.8

### Packages
```
python -m venv venv
. ./venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
(optional) tensorboard --logidr runs/run_label
```
### Data

* Original datasets (MNIST, FASHION, SVHN, IMAGENET, CIFAR) will be downloaded automatically.
* Pre-trained features for IMAGENET200 and CIFAR100 can be downloaded from: https://drive.google.com/drive/folders/1gM5rgK-_HS-S1M2LzYRcofXRENieoVbg?usp=sharing
* The extracted features should be put in: ***pytorch_data/extracted***.

## Experiments
* Different configurations for MIX: ``` python mix_run.py -t params -p all -d all -rl mix-params -dev cuda:0 ```
* Final versions (*benchmark/mix/mix_final_exp.py*) for MIX: ``` python mix_run.py -t final -v all -d all -rl mix-final -dev cuda:0  ```
* Baselines: 
  * ```python mix_run.py -t baselines -b naive er ersb agem der -lr 0.0001 -bn 0 -d all -rl mix-baselines -dev cuda:0```
  * ```python mix_run.py -t baselines -b icarl gss lwf si -lr 0.0001 -bn 1 -d all -rl mix-baselines -dev cuda:0```

## Results

* Output directory: ***results***.
* [optional] Post-processing (set your paths in the script): ``` python -m scripts.mix_results_proc ```
* [optional] Tensorboard: *http://localhost:6006/*

