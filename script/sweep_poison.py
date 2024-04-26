from pathlib import Path
import os


def get_extractor(dataset, ssl_method):
    path = Path("my_zoo") / Path(dataset) / Path(ssl_method) / Path("SimLFB")
    while os.path.isdir(path):
        path = path / os.listdir(path)[0]
    return str(path)


def main():

    for dataset in ["imagenet100"]:
        for poison_method in ["knn"]:
            for ssl_method in ["simclr"]:
                for seed in range(56, 60):
                    arg_list = [
                        f"CUDA_VISIBLE_DEVICES=1",
                        "python",
                        "main_4poison_new.py",
                        "--dataset",
                        str(dataset),
                        "--backbone",
                        "resnet18",
                        "--data_dir",
                        str("../data/" + dataset),
                        "--train_dir",
                        "train",
                        "--optimizer",
                        "sgd",
                        "--pretrained_feature_extractor",
                        str(get_extractor(dataset, ssl_method)),
                        "--poison_rate",
                        "0.6",
                        "--poison_method",
                        str(poison_method),
                        "--pretrain_method",
                        str(ssl_method),
                        "--random_seed",
                        str(seed)
                    ]
                    command = ' '.join(arg_list)

                    print(command)

                    os.system(command=command)


main()
