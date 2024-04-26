from torchvision.datasets import CIFAR10
from poisoning_utils import split_cifar, get_trigger, add_trigger
from PIL import Image
import torch

train_dataset = CIFAR10(
    "./datasets",
    train=True,
    download=True,
    transform=None,
)

train_dataset = split_cifar(train_dataset, "cifar10", True)

pattern, mask = get_trigger("cifar10", "gaussian_noise")

train_dataset.data = train_dataset.data[torch.tensor(train_dataset.targets) == 7]

train_dataset.data = add_trigger(
                train_dataset.data,
                pattern,
                mask,
                trigger_alpha=0.2,
            )

Image.fromarray(train_dataset.data[5]).save("image_example/" + "cifar10_hellokitty.png")
