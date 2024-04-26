CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-clb-None-47-0.200-0-1.0000         >    log/cifar10/simclr/cifar10-resnet18-clb-None-47-0.200-0-1.0000.log       &
CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-clb-None-47-0.400-0-1.0000         >    log/cifar10/simclr/cifar10-resnet18-clb-None-47-0.400-0-1.0000.log       &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-clb-None-47-0.800-0-1.0000         >    log/cifar10/simclr/cifar10-resnet18-clb-None-47-0.800-0-1.0000.log       &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-clb-None-47-1.000-0-1.0000         >    log/cifar10/simclr/cifar10-resnet18-clb-None-47-1.000-0-1.0000.log       &

CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-47-0.200-3-0.9922    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-0.200-3-0.9922.log  &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-47-0.400-7-0.9350    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-0.400-7-0.9350.log  &
CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-47-0.800-1-0.9567    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-0.800-1-0.9567.log  &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-47-1.000-1-0.9062    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-1.000-1-0.9062.log  &

CUDA_VISIBLE_DEVICES=6 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-47-0.200-8-0.9167       >    log/cifar10/simclr/cifar10-resnet18-knn-simclr-47-0.200-8-0.9167.log     &
CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-47-0.400-8-0.9167       >    log/cifar10/simclr/cifar10-resnet18-knn-simclr-47-0.400-8-0.9167.log     &
CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-47-0.800-6-0.9497       >    log/cifar10/simclr/cifar10-resnet18-knn-simclr-47-0.800-6-0.9497.log     &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-47-1.000-5-0.6258       >    log/cifar10/simclr/cifar10-resnet18-knn-simclr-47-1.000-5-0.6258.log     &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-43-0.400-7-0.9350    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-43-0.400-7-0.9350.log  &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-43-0.600-7-0.8307    >    log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-43-0.600-7-0.8307.log  &

