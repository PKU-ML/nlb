CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log/cifar100/simclr/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log/cifar100/simclr/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar100/simclr/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100/mocov2plus.sh  cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log/cifar100/mocov2plus/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar100/mocov2plus.sh  cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log/cifar100/mocov2plus/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100/mocov2plus.sh  cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar100/mocov2plus/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100/byol.sh        cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log/cifar100/byol/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar100/byol.sh        cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log/cifar100/byol/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100/byol.sh        cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar100/byol/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar100/barlow.sh      cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074 > log/cifar100/barlow/cifar100-resnet18-dcsnew-simclr-43-0.600-41-0.9074.log &
CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar100/barlow.sh      cifar100-resnet18-knn-simclr-49-0.600-39-0.9593 > log/cifar100/barlow/cifar100-resnet18-knn-simclr-49-0.600-39-0.9593.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar100/barlow.sh      cifar100-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar100/barlow/cifar100-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-random-None-35-0.600-32-0.0222 > log/cifar100/simclr/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar100/mocov2plus.sh  cifar100-resnet18-random-None-35-0.600-32-0.0222 > log/cifar100/mocov2plus/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &
CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar100/byol.sh        cifar100-resnet18-random-None-35-0.600-32-0.0222 > log/cifar100/byol/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar100/barlow.sh      cifar100-resnet18-random-None-35-0.600-32-0.0222 > log/cifar100/barlow/cifar100-resnet18-random-None-35-0.600-32-0.0222.log &

CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-clb-None-3941-0.600-39-1.0000 > log/cifar100/simclr/cifar100-resnet18-clb-None-3941-0.600-39-1.0000.log &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-clb-None-4140-0.600-41-1.0000 > log/cifar100/simclr/cifar100-resnet18-clb-None-4140-0.600-41-1.0000.log &

CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-random-None-34-0.600-72-0.0296 > log/cifar100/simclr/cifar100-resnet18-random-None-34-0.600-72-0.0296.log &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar100/simclr.sh      cifar100-resnet18-random-None-36-0.600-5-0.0259  > log/cifar100/simclr/cifar100-resnet18-random-None-36-0.600-5-0.0259.log &
