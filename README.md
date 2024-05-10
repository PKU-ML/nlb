## <strong><span style="color: red;">Note: Code is still incomplete, please wait a moment, thank you!</span></strong>

# No Label Backdoor

This is the official repository for "[How to Craft Backdoors with Unlabeled Data Alone?](https://arxiv.org/abs/2404.06694)" by Yifei Wang*, Wenhan Ma*, Stefanie Jegelka, Yisen Wang.

![illustration](assets/illustration.png)

## Requirements

```
torch==1.10.0
torchvision==0.11.1
einops
pytorch-lightning==1.5.3
torchmetrics==0.6.0
lightning-bolts>=0.4.0
tqdm
wandb
scipy
timm
```

## Run on CIFAR-10

### Step 1:

Split training set into pretraining set and downstream set. Run
`python ./misc/cifar10_split.py`

### Step 2:

Pretrain the clean encoder on pretraining set. Run the script in `script/cifar10_encoder`

### Step 3:

Select the poison subset. Run the script in `script/cifar10_poison`

### Step 4:

Pretrain the backdoor encoder on pretraining set with poison subset. Run the script in `script/cifar10_pretrain`

### Step 5:

Train the classifier. Run the script in `script/cifar10_linear`

## Run on ImageNet-100

### Step 1:

Split training set into pretraining set and downstream set. Run

`python ./misc/imagnet_script/in100_split.py`

Create a copy with poison. Run

`python ./misc/imagnet_script/in100_add_trigger.py`

Create a folder with soft link to the clean dataset. Run

`python misc/imagnet_script/in100_link.py`

### Step 2:

Pretrain the clean encoder on pretraining set. Run the script in `script/cifar10_encoder`

### Step 3:

Select the poison subset. Run the script in `script/cifar10_poison`

### Step 4:

Create a folder with soft link to the clean dataset and poison subset. Run

`python misc/imagnet_script/in100_link_poison.py`

### Step 5:

Pretrain the backdoor encoder on pretraining set with poison subset. Run the script in `script/cifar10_pretrain`

### Step 6:

Train the classifier. Run the script in `script/cifar10_linear`

## Citation

If you find this useful in your research, please consider citing:

```
@misc{wang2024craft,
      title={How to Craft Backdoors with Unlabeled Data Alone?}, 
      author={Yifei Wang and Wenhan Ma and Stefanie Jegelka and Yisen Wang},
      year={2024},
      eprint={2404.06694},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement

This repo is based upon the following repository:

* [solo-learn](https://github.com/vturrisi/solo-learn)
