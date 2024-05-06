import random
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


rate = 0.1
CLASS = CIFAR10
class_num = 10


def main():

    random.seed(42)

    dataset = CLASS(root="./cifar10/train",
                    train=True, download=True)

    lens = len(dataset)
    split_index = int((lens / class_num) * (1-rate))

    lst_train = []

    for i in range(class_num):
        lst_train.append([])

    for j in range(lens):
        lst_train[dataset.targets[j]].append(j)

    lst_pre, lst_down = [], []

    for k in range(class_num):
        random.shuffle(lst_train[k])
        lst_pre.extend(lst_train[k][:split_index])
        lst_down.extend(lst_train[k][split_index:])

    random.shuffle(lst_pre)
    random.shuffle(lst_down)

    lst_pre = np.array(lst_pre, dtype=int)
    lst_down = np.array(lst_down, dtype=int)

    np.savetxt("./cifar10_pre.txt", lst_pre, "%d")
    np.savetxt("./cifar10_down.txt", lst_down, "%d")


if __name__ == "__main__":
    main()
