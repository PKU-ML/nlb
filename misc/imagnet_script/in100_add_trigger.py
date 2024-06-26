import argparse
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path
import numpy as np
import os
import random
import tqdm

patch = Image.open("misc/trigger_imnet.png")

random.seed(42)


def add_trigger(
    sample: Image.Image,
    trigger: str,
    K: float
) -> Image.Image:
    if trigger == "patch":
        sample_array = np.array(sample)
        shape = (
            int(sample_array.shape[0] / K),
            int(sample_array.shape[1] / K)
        )
        place = (
            random.randint(int(shape[0]), int(shape[0]*(K-2))),
            random.randint(int(shape[1]), int(shape[1]*(K-2)))
        )
        sample_array[place[0]:place[0]+shape[0],
                     place[1]:place[1] + shape[1]] = \
            np.array(patch.resize((shape[1], shape[0])))
        return Image.fromarray(sample_array)

    else:
        assert trigger == "patch"


def posion(normal_dir, poison_dir, K):

    os.makedirs(poison_dir, exist_ok=True)

    normal_dataloader = ImageFolder(normal_dir)
    size = len(normal_dataloader)

    for i in tqdm.tqdm(range(size)):
        img = normal_dataloader[i][0]
        normal_path = normal_dataloader.imgs[i][0]
        poison_path = poison_dir / \
            Path(normal_path[len(str(normal_dir))+1:])
        newimg = add_trigger(img, "patch", K)
        os.makedirs(poison_path.parent, exist_ok=True)
        newimg.save(poison_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", type=Path,
                        default="./data/imagenet100")
    args = parser.parse_args()

    posion(args.imagenet_dir / "train",
           args.imagenet_dir / "train_poison")
    posion(args.imagenet_dir / "linear",
           args.imagenet_dir / "linear_poison")
    posion(args.imagenet_dir / "val",
           args.imagenet_dir / "val_poison")


if __name__ == "__main__":
    main()
