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

    os.makedirs(newlink_dir, exist_ok=False)
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
                        default="../data/imagenet100")
    # parser.add_argument("--orilink_dir", type=Path,
    #                     default="datasets/imagenet100/clean_pre_link")
    # parser.add_argument("--poison_pre_dir", type=Path,
    #                     default="datasets/imagenet100/poison_pre")
    # parser.add_argument("--poison_res_dir", type=Path,
    #                     default="datasets/imagenet100/poison")
    parser.add_argument("--poison_name", type=str,
                        default="imagenet100_resnet18_dcs3_simclr_rate_0.750_gaussian_noise_alpha_0.20_trial_0_class_20_acc_0.9754_dis_0.8519")
    args = parser.parse_args()
    relink(args.imagenet_dir, "imagenet100-resnet18-knn-simclr-50-0.600-22-0.8234")
    relink(args.imagenet_dir, "imagenet100-resnet18-dcs-simclr-40-0.600-48-0.7350")
    relink(args.imagenet_dir, "imagenet100-resnet18-dcsnew-simclr-40-0.600-80-0.9843")
    relink(args.imagenet_dir, "imagenet100-resnet18-clb-None-0-0.600-0-1.0000")
    relink(args.imagenet_dir, "imagenet100-resnet18-knn-simclr-52-0.600-91-0.9217")


if __name__ == "__main__":
    main()
