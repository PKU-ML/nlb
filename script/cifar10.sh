CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/mocov2plus.sh  cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/mocov2plus/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar10/byol.sh        cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/byol/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/barlow.sh      cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/barlow/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &

CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-50-0.600-0-0.8693 > log/cifar10/simclr/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/mocov2plus.sh  cifar10-resnet18-knn-simclr-50-0.600-0-0.8693 > log/cifar10/mocov2plus/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar10/byol.sh        cifar10-resnet18-knn-simclr-50-0.600-0-0.8693 > log/cifar10/byol/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/barlow.sh      cifar10-resnet18-knn-simclr-50-0.600-0-0.8693 > log/cifar10/barlow/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-42-0.600-0-1.0000.log &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/mocov2plus.sh  cifar10-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar10/mocov2plus/cifar10-resnet18-clb-None-42-0.600-0-1.0000.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar10/byol.sh        cifar10-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar10/byol/cifar10-resnet18-clb-None-42-0.600-0-1.0000.log &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/barlow.sh      cifar10-resnet18-clb-None-42-0.600-0-1.0000 > log/cifar10/barlow/cifar10-resnet18-clb-None-42-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-random-None-35-0.600-4-0.1070 > log/cifar10/simclr/cifar10-resnet18-random-None-35-0.600-4-0.1070 &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/mocov2plus.sh  cifar10-resnet18-random-None-35-0.600-4-0.1070 > log/cifar10/mocov2plus/cifar10-resnet18-random-None-35-0.600-4-0.1070 &
CUDA_VISIBLE_DEVICES=6 nohup bash script/cifar10/byol.sh        cifar10-resnet18-random-None-35-0.600-4-0.1070 > log/cifar10/byol/cifar10-resnet18-random-None-35-0.600-4-0.1070 &
CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/barlow.sh      cifar10-resnet18-random-None-35-0.600-4-0.1070 > log/cifar10/barlow/cifar10-resnet18-random-None-35-0.600-4-0.1070 &



CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-clb-None-742-0.600-7-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-742-0.600-7-1.0000.log &

CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcs-simclr-43-0.600-1-0.9904 > log/cifar10/simclr/cifar10-resnet18-dcs-simclr-43-0.600-1-0.9904.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcs-simclr-47-0.600-6-0.9833 > log/cifar10/simclr/cifar10-resnet18-dcs-simclr-47-0.600-6-0.9833.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-46-0.600-4-0.2467 > log/cifar10/simclr/cifar10-resnet18-knn-simclr-46-0.600-4-0.2467.log &
CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-knn-simclr-54-0.600-0-0.4548 > log/cifar10/simclr/cifar10-resnet18-knn-simclr-54-0.600-0-0.4548.log &

CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsneg-simclr-43-0.600-1-0.8407 > log/cifar10/simclr/cifar10-resnet18-dcsneg-simclr-43-0.600-1-0.8407.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsneg-simclr-47-0.600-2-0.3911 > log/cifar10/simclr/cifar10-resnet18-dcsneg-simclr-47-0.600-2-0.3911.log &
CUDA_VISIBLE_DEVICES=6 nohup bash script/cifar10/simclr.sh      cifar10-resnet18-dcsneg-simclr-51-0.600-1-0.6274 > log/cifar10/simclr/cifar10-resnet18-dcsneg-simclr-51-0.600-1-0.6274.log &



## 附加实验

CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/dino.sh        cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/dino/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/dino.sh        cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/dino/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log &

CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10/mocov3.sh        cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/mocov3/cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/mocov3.sh        cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/mocov3/cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log &


CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-clb-None-44-0.600-3-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-44-0.600-3-1.0000.log &
CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-clb-None-43-0.600-2-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-43-0.600-2-1.0000.log &
CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-clb-None-42-0.600-1-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-42-0.600-1-1.0000.log &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-clb-None-41-0.600-0-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-41-0.600-0-1.0000.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-dcsnew-simclr-251-0.100-9-0.9933 > log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-251-0.100-9-0.9933.log &
CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-dcsnew-simclr-247-0.100-2-0.9956 > log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-247-0.100-2-0.9956.log &
CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-dcsnew-simclr-151-0.500-7-0.9204 > log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-151-0.500-7-0.9204.log &
CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-dcsnew-simclr-147-0.500-7-0.9369 > log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-147-0.500-7-0.9369.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-knn-simclr-147-0.500-1-0.9862    > log/cifar10/simclr/cifar10-resnet18-knn-simclr-147-0.500-1-0.9862.log &
CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-knn-simclr-247-0.100-2-0.9044    > log/cifar10/simclr/cifar10-resnet18-knn-simclr-247-0.100-2-0.9044.log &

CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/simclr.sh        cifar10-resnet18-infonce-simclr-43-0.600-7-0.8930 > log/cifar10/simclr/cifar10-resnet18-infonce-simclr-43-0.600-7-0.8930.log &

