CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100_linear/simclr.sh          checkpoint/simclr-cifar100_poison_cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074-checkerboard_center-1.0/1eyv8bia/simclr-cifar100-1eyv8bia-ep=499.ckpt                 cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log_linear/cifar100/simclr/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100_linear/simclr.sh          checkpoint/simclr-cifar100_poison_cifar100-resnet18-knn-simclr-49-0.600-39-0.9593-checkerboard_center-1.0/fe02ig6q/simclr-cifar100-fe02ig6q-ep=499.ckpt                    cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log_linear/cifar100/simclr/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar100_linear/simclr.sh          checkpoint/simclr-cifar100_poison_cifar100-resnet18-clb-None-42-0.600-0-1.0000-checkerboard_center-1.0/x8emmkj3/simclr-cifar100-x8emmkj3-ep=499.ckpt                       cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log_linear/cifar100/simclr/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100_linear/mocov2plus.sh      checkpoint/mocov2plus-cifar100_poison_cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074-checkerboard_center-1.0/ovk0ycqy/mocov2plus-cifar100-ovk0ycqy-ep=499.ckpt         cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log_linear/cifar100/mocov2plus/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100_linear/mocov2plus.sh      checkpoint/mocov2plus-cifar100_poison_cifar100-resnet18-knn-simclr-49-0.600-39-0.9593-checkerboard_center-1.0/vn1i86mp/mocov2plus-cifar100-vn1i86mp-ep=499.ckpt            cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log_linear/cifar100/mocov2plus/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar100_linear/mocov2plus.sh      checkpoint/mocov2plus-cifar100_poison_cifar100-resnet18-clb-None-42-0.600-0-1.0000-checkerboard_center-1.0/yowxe7uq/mocov2plus-cifar100-yowxe7uq-ep=499.ckpt               cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log_linear/cifar100/mocov2plus/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100_linear/byol.sh            checkpoint/byol-cifar100_poison_cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074-checkerboard_center-1.0/crid268n/byol-cifar100-crid268n-ep=499.ckpt                     cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log_linear/cifar100/byol/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100_linear/byol.sh            checkpoint/byol-cifar100_poison_cifar100-resnet18-knn-simclr-49-0.600-39-0.9593-checkerboard_center-1.0/5uip648r/byol-cifar100-5uip648r-ep=499.ckpt                        cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log_linear/cifar100/byol/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar100_linear/byol.sh            checkpoint/byol-cifar100_poison_cifar100-resnet18-clb-None-42-0.600-0-1.0000-checkerboard_center-1.0/e1c76weu/byol-cifar100-e1c76weu-ep=499.ckpt                           cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log_linear/cifar100/byol/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100_linear/barlow.sh          checkpoint/barlow-cifar100_poison_cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074-checkerboard_center-1.0/fcxeiq6z/barlow-cifar100-fcxeiq6z-ep=499.ckpt                 cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log_linear/cifar100/barlow/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100_linear/barlow.sh          checkpoint/barlow-cifar100_poison_cifar100-resnet18-knn-simclr-49-0.600-39-0.9593-checkerboard_center-1.0/zvicwyj5/barlow-cifar100-zvicwyj5-ep=499.ckpt                    cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log_linear/cifar100/barlow/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar100_linear/barlow.sh          checkpoint/barlow-cifar100_poison_cifar100-resnet18-clb-None-42-0.600-0-1.0000-checkerboard_center-1.0/evsa35e8/barlow-cifar100-evsa35e8-ep=499.ckpt                       cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log_linear/cifar100/barlow/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar100_linear/simclr.sh          checkpoint/simclr-cifar10_poison_cifar10-resnet18-random-None-35-0.600-4-0.1070-checkerboard_center-1.0/nvkax5kx/simclr-cifar10-nvkax5kx-ep=499.ckpt                       cifar100-resnet18-random-None-35-0.600-32-0.0222 > log_linear/cifar100/simclr/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &
CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100_linear/mocov2plus.sh      checkpoint/mocov2plus-cifar100_poison_cifar100-resnet18-random-None-35-0.600-32-0.0222-checkerboard_center-1.0/fldxq7qm/mocov2plus-cifar100-fldxq7qm-ep=499.ckpt           cifar100-resnet18-random-None-35-0.600-32-0.0222 > log_linear/cifar100/mocov2plus/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar100_linear/byol.sh            checkpoint/byol-cifar100_poison_cifar100-resnet18-random-None-35-0.600-32-0.0222-checkerboard_center-1.0/7ah0xhfg/byol-cifar100-7ah0xhfg-ep=499.ckpt                       cifar100-resnet18-random-None-35-0.600-32-0.0222 > log_linear/cifar100/byol/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar100_linear/barlow.sh          checkpoint/barlow-cifar100_poison_cifar100-resnet18-random-None-35-0.600-32-0.0222-checkerboard_center-1.0/03aas8pj/barlow-cifar100-03aas8pj-ep=499.ckpt                   cifar100-resnet18-random-None-35-0.600-32-0.0222 > log_linear/cifar100/barlow/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &

CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar100_linear/simclr.sh          checkpoint/simclr-cifar100_poison_cifar100-resnet18-clb-None-3941-0.600-39-1.0000-checkerboard_center-1.0/p557tkx3/simclr-cifar100-p557tkx3-ep=499.ckpt                    cifar100-resnet18-clb-None-3941-0.600-39-1.0000 > log_linear/cifar100/simclr/cifar100-resnet18-clb-None-3941-0.600-39-1.0000.log &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar100_linear/simclr.sh          checkpoint/simclr-cifar100_poison_cifar100-resnet18-clb-None-4140-0.600-41-1.0000-checkerboard_center-1.0/ioz8ikay/simclr-cifar100-ioz8ikay-ep=499.ckpt                    cifar100-resnet18-clb-None-4140-0.600-41-1.0000 > log_linear/cifar100/simclr/cifar100-resnet18-clb-None-4140-0.600-41-1.0000.log &
