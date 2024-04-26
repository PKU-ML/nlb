CUDA_VISIBLE_DEVICES=0 nohup bash script/weight_decay/weight_decay3.sh      cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=0 nohup bash script/weight_decay/weight_decay3.sh      cifar10-resnet18-knn-simclr-50-0.600-0-0.8693       >    log/cifar10/simclr/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=6 nohup bash script/weight_decay/weight_decay3.sh      cifar10-resnet18-clb-None-42-0.600-0-1.0000         >    log/cifar10/simclr/cifar10-resnet18-clb-None-42-0.600-0-1.0000.log      &

CUDA_VISIBLE_DEVICES=7 nohup bash script/weight_decay/weight_decay4.sh      cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=8 nohup bash script/weight_decay/weight_decay4.sh      cifar10-resnet18-knn-simclr-50-0.600-0-0.8693       >    log/cifar10/simclr/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=9 nohup bash script/weight_decay/weight_decay4.sh      cifar10-resnet18-clb-None-42-0.600-0-1.0000         >    log/cifar10/simclr/cifar10-resnet18-clb-None-42-0.600-0-1.0000.log      &
