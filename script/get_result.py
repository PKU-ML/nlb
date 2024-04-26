import os

lst = [
"log_linear/cifar10/simclr/cifar10-resnet18-dcsnew-barlow_twins-75-0.600-7-0.8304.log",
"log_linear/cifar10/simclr/cifar10-resnet18-dcsnew-byol-71-0.600-2-0.8815.log",
"log_linear/cifar10/simclr/cifar10-resnet18-dcsnew-mocov2plus-67-0.600-6-0.9537.log",
"log_linear/cifar10/simclr/cifar10-resnet18-knn-barlow_twins-78-0.600-2-0.6544.log",
"log_linear/cifar10/simclr/cifar10-resnet18-knn-byol-74-0.600-7-0.9737.log",
"log_linear/cifar10/simclr/cifar10-resnet18-knn-mocov2plus-68-0.600-7-0.9463.log",
]

report = [
    "clean_val_acc1",
    "poison_val_acc1",
    "poison_nfp",
    "poison_val_asr",
]


for filename in lst:
    f = open(filename)
    str = f.read()
    print(filename, end=" ")
    for name in report:
        start = str.rfind(name)
        end = str[start:].find('\n')
        print(str[start:end+start], end=' ')
    print("")
    f.close()
