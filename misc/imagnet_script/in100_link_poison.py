import argparse
from torchvision.datasets import ImageFolder
from pathlib import Path
import os
import torch
import tqdm


def relink(imagenet_dir: Path, poison_name: Path):

    orilink_dir: Path = imagenet_dir / "train_link"
    newlink_dir: Path = imagenet_dir / "poison" / poison_name
    poison_dir: Path = imagenet_dir / "train_poison"
    pt_file: Path = imagenet_dir / "poison" / (str(poison_name) + ".pt")

    os.makedirs(newlink_dir, exist_ok=True)
    os.removedirs(newlink_dir)
    os.system(f"cp -r {orilink_dir} {newlink_dir}")
    dataloader = ImageFolder(newlink_dir)
    poison_pt = torch.load(pt_file)
    for i in tqdm.tqdm(poison_pt["poisoning_index"]):
        path = dataloader.imgs[i][0]
        sub_file_name = path[len(str(newlink_dir))+1:]
        poi_path = poison_dir / sub_file_name
        os.system(f"rm {path}")
        os.system(f"ln -s {poi_path.absolute()} {path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", type=Path,
                        default="./data/imagenet100")
    parser.add_argument("--poison_info", type=Path,
                        default="imagenet100-resnet18-clb-None-11-0.600-1-1.0000")
    args = parser.parse_args()

    relink(args.imagenet_dir, args.poison_info)

if __name__ == "__main__":
    main()
