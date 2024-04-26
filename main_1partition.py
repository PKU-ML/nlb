import argparse
from pathlib import Path
import os
import random
import shutil
import tqdm


def partition(train_dir: Path, linear_dir: Path, ratio: float):

    os.makedirs(linear_dir, exist_ok=False)

    for subfolder in tqdm.tqdm(os.listdir(train_dir)):

        train_folder = train_dir / Path(subfolder)
        linear_folder = linear_dir / Path(subfolder)
        os.makedirs(linear_folder, exist_ok=False)

        file_list = os.listdir(train_folder)
        subset_len = int(len(file_list) * ratio)
        subset_file_list = random.sample(file_list, subset_len)

        for file_name in subset_file_list:
            shutil.move(train_folder / file_name, linear_folder / file_name)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path,
                        default="../data/imagenet100/train")
    parser.add_argument("--linear", type=Path,
                        default="../data/imagenet100/linear")
    parser.add_argument("--ratio", type=float, default="0.1")
    args = parser.parse_args()
    partition(args.train, args.linear, 0.1)


if __name__ == "__main__":
    main()
