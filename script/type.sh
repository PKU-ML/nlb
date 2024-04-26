CUDA_VISIBLE_DEVICES=0 nohup bash script/type/checkerboard_1corner.sh      cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/c1_cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=1 nohup bash script/type/checkerboard_4corner.sh      cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/c4_cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=4 nohup bash script/type/checkerboard_full.sh         cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/cf_cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=6 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/gn_cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=6 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/gn2_cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &
CUDA_VISIBLE_DEVICES=1 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041 > log/cifar10/simclr/gn3_cifar10-resnet18-dcsnew-simclr-47-0.600-7-0.9041.log &

CUDA_VISIBLE_DEVICES=0 nohup bash script/type/checkerboard_1corner.sh      cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/simclr/c1_cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=1 nohup bash script/type/checkerboard_4corner.sh      cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/simclr/c4_cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=4 nohup bash script/type/checkerboard_full.sh         cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/simclr/cf_cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=6 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/simclr/gn_cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=6 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/simclr/gn2_cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &
CUDA_VISIBLE_DEVICES=6 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-knn-simclr-50-0.600-0-0.8693    > log/cifar10/simclr/gn3_cifar10-resnet18-knn-simclr-50-0.600-0-0.8693.log    &

CUDA_VISIBLE_DEVICES=9 nohup bash script/type/gaussian_noise.sh            cifar10-resnet18-random-None-35-0.600-4-0.1070   > log/cifar10/simclr/gn3_cifar10-resnet18-random-None-35-0.600-4-0.1070.log   &
