import random
import numpy as np
from torchvision.datasets import CIFAR10


downstream_rate = 0.1
CLASS = CIFAR10
class_num = 10
seed = 42

def main():

    random.seed(seed)

    dataset = CIFAR10(root="./data/cifar10",
                      train=True, download=True)

    lens = len(dataset)
    split_index = int((lens / class_num) * (1-downstream_rate))

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

    np.savetxt("./misc/cifar10_pre.txt", lst_pre, "%d")
    np.savetxt("./misc/cifar10_down.txt", lst_down, "%d")


if __name__ == "__main__":
    main()
