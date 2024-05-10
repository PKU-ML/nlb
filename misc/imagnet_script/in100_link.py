import argparse
from pathlib import Path
import os
import tqdm


def link(target_dir: Path, link_dir: Path):

    os.makedirs(link_dir, exist_ok=False)

    for folder in tqdm.tqdm(os.listdir(target_dir)):

        os.makedirs(link_dir / Path(folder), exist_ok=False)

        for filename in tqdm.tqdm(os.listdir(target_dir / Path(folder))):

            target_path = (target_dir / Path(folder) / filename).absolute()
            link_path = link_dir / Path(folder) / filename

            os.system(f"ln -s {target_path} {link_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", type=Path,
                        default="./data/imagenet100")
    args = parser.parse_args()

    link(args.imagenet_dir / "train",
         args.imagenet_dir / "train_link")
    link(args.imagenet_dir / "linear",
         args.imagenet_dir / "linear_link")


if __name__ == "__main__":
    main()
