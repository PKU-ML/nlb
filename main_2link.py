import argparse
from pathlib import Path
import os
import tqdm


def link(normal_dir: Path, orilink_dir: Path):

    os.makedirs(orilink_dir, exist_ok=False)

    for folder in tqdm.tqdm(os.listdir(normal_dir)):

        os.makedirs(orilink_dir / Path(folder), exist_ok=False)

        for filename in os.listdir(normal_dir / Path(folder)):

            target_path = (normal_dir / Path(folder) / filename).absolute()
            link_path = orilink_dir / Path(folder) / filename

            os.system(f"ln -s {target_path} {link_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", type=Path,
                        default="../data/imagenet100")
    args = parser.parse_args()

    link(args.imagenet_dir / "train",
         args.imagenet_dir / "train_link")
    link(args.imagenet_dir / "linear",
         args.imagenet_dir / "linear_link")


if __name__ == "__main__":
    main()
