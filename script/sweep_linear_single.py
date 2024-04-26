import os
import sys
import time

lens = 1
no = 0#int(sys.argv[1])
cuda = [1,2,3,4,6,7,8]
run = False
command_list = []

for dataset in ["cifar10"]:#, "cifar100"
    for ssl_method in ["byol", "barlow", "simclr", "mocov2plus"]:#
        bash_file = "script/" + dataset + '_linear/' + ssl_method + ".sh"
        if dataset == "cifar10":
            poison_data_list = [
                "cifar10-resnet18-clb-None-17-0.600-0-1.0000",
                # "cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681",
                # "cifar10-resnet18-knn-simclr-16-0.600-4-0.7422"
            ]
        elif dataset == "cifar100":
            poison_data_list = [
                # "cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111",
                # "cifar100-resnet18-knn-simclr-40-0.600-53-0.7074",
                # "cifar100-resnet18-clb-None-0-0.600-0-1.0000"
            ]
        for poison_data in poison_data_list:
            folder_list = os.listdir("checkpoint")
            folder_list = [name for name in folder_list if ssl_method + '-' + dataset + '_' in name and poison_data in name]
            assert(len(folder_list)==1)
            folder_name = "checkpoint/" + folder_list[0]
            folder_name = folder_name + '/' + os.listdir(folder_name)[0]
            flist = os.listdir(folder_name)
            flist.remove("args.json")
            assert(len(flist)==1)
            pretrained_feature_extractor = folder_name + '/' + flist[0]
            os.makedirs("log_linear/" + dataset + "/" + ssl_method, exist_ok=True)
            log_file = "log_linear/" + dataset + "/" + ssl_method + '/' + poison_data + ".log"
            command_list.append("bash"+" " + bash_file +
                                " " + pretrained_feature_extractor + " " + poison_data + " > " + log_file)

new_list = ["CUDA_VISIBLE_DEVICES=" +
            str(cuda[no]) + ' ' + i for i in command_list[no::lens]]
for e in new_list:
    print(e,flush=True)
    # time.sleep(no)
    if run:
        os.system(e)
