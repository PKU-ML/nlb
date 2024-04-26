import random
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


def split(dataset, class_num, rate=0.1):

    lens = len(dataset)
    split_index = int((lens / class_num) * (1-rate))
    split_index_r = int((lens / class_num) * rate)

    lst_train = []

    for i in range(class_num):
        lst_train.append([])

    for j in range(lens):
        lst_train[dataset.targets[j]].append(j)

    lst_pre, lst_down, lst_down2 = [], [], []

    for k in range(class_num):
        random.shuffle(lst_train[k])
        lst_pre.extend(lst_train[k][:split_index])
        lst_down.extend(lst_train[k][split_index:])
        lst_down2.extend(lst_train[k][split_index - split_index_r:split_index])

    random.shuffle(lst_pre)
    random.shuffle(lst_down)
    random.shuffle(lst_down2)

    lst_pre = np.array(lst_pre, dtype=int)
    lst_down = np.array(lst_down, dtype=int)
    lst_down2 = np.array(lst_down2, dtype=int)

    return lst_pre, lst_down, lst_down2


def main():

    random.seed(42)

    cifar10 = CIFAR10(root="../data/cifar10/train",
                      train=True, download=True)
    cifar100 = CIFAR100(root="../data/cifar100/train",
                        train=True, download=True)

    pre_cifar10, down_cifar10, down2_cifar10 = split(cifar10, 10)
    pre_cifar100, down_cifar100, down2_cifar100 = split(cifar100, 100)

    np.savetxt("./data/cifar10_pre.txt",   pre_cifar10,    "%d")
    np.savetxt("./data/cifar10_down.txt",  down_cifar10,   "%d")
    np.savetxt("./data/cifar10_down2.txt", down2_cifar10,  "%d")

    np.savetxt("./data/cifar100_pre.txt",  pre_cifar100,   "%d")
    np.savetxt("./data/cifar100_down.txt", down_cifar100,  "%d")
    np.savetxt("./data/cifar100_down2.txt", down2_cifar100, "%d")


def test():

    pre_cifar10   = np.loadtxt("./data/cifar10_pre.txt",    dtype=int)
    down_cifar10  = np.loadtxt("./data/cifar10_down.txt",   dtype=int)
    down2_cifar10 = np.loadtxt("./data/cifar10_down2.txt",  dtype=int)

    pre_cifar100  = np.loadtxt("./data/cifar100_pre.txt",   dtype=int)
    down_cifar100 = np.loadtxt("./data/cifar100_down.txt",  dtype=int)
    down2_cifar100 = np.loadtxt("./data/cifar100_down2.txt", dtype=int)

    cifar10 = CIFAR10(root="../data/cifar10/train",
                      train=True, download=True)
    cifar100 = CIFAR100(root="../data/cifar100/train",
                        train=True, download=True)
    for i in range(10):
        a = (np.array(cifar100.targets)[down2_cifar100] == i).sum()
        print(a)


main()
