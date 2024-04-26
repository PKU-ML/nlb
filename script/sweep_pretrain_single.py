import os
import sys
import time

lens = 1
no = 0
cuda = [1, 6, 7, 8]
run = False
command_list = []

for dataset in ["cifar10", "cifar100"]: #
    for ssl_method in ["simclr", "mocov2plus", "byol", "barlow"]:#
        bash_file = "script/" + dataset + '/' + ssl_method + ".sh"
        if dataset == "cifar10":
            poison_data_list = [
                "cifar10-resnet18-clb-None-17-0.600-0-1.0000",
                "cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681",
                "cifar10-resnet18-knn-simclr-16-0.600-4-0.7422"
            ]
        elif dataset == "cifar100":
            poison_data_list = [
                "cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111",
                "cifar100-resnet18-knn-simclr-40-0.600-53-0.7074",
                "cifar100-resnet18-clb-None-0-0.600-0-1.0000"
            ]
        for poison_data in poison_data_list:
            os.makedirs("log/" + dataset + "/" + ssl_method, exist_ok=True)
            log_file = "log/" + dataset + "/" + ssl_method + '/' + poison_data + ".log"
            command_list.append("bash"+" " + bash_file +
                                " " + poison_data + " > " + log_file)

new_list = ["CUDA_VISIBLE_DEVICES=" +
            str(cuda[no]) + ' ' + i for i in command_list[no::lens]]
for e in new_list:
    print(e,flush=True)
    time.sleep(no)
    if run:
        os.system(e)
