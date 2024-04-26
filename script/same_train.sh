CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/mocov2plus.sh  cifar10-resnet18-dcsnew-mocov2plus-67-0.600-6-0.9537       > log/cifar10/mocov2plus/cifar10-resnet18-dcsnew-mocov2plus-67-0.600-6-0.9537       &
CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/byol.sh        cifar10-resnet18-dcsnew-byol-71-0.600-2-0.8815             > log/cifar10/byol/cifar10-resnet18-dcsnew-byol-71-0.600-2-0.8815                   &
CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/barlow.sh      cifar10-resnet18-dcsnew-barlow_twins-75-0.600-7-0.8304     > log/cifar10/barlow/cifar10-resnet18-dcsnew-barlow_twins-75-0.600-7-0.8304         &

CUDA_VISIBLE_DEVICES=6 nohup bash script/cifar10/mocov2plus.sh  cifar10-resnet18-knn-mocov2plus-68-0.600-7-0.9463          > log/cifar10/mocov2plus/cifar10-resnet18-knn-mocov2plus-68-0.600-7-0.9463          &
CUDA_VISIBLE_DEVICES=6 nohup bash script/cifar10/byol.sh        cifar10-resnet18-knn-byol-74-0.600-7-0.9737                > log/cifar10/byol/cifar10-resnet18-knn-byol-74-0.600-7-0.9737                      &
CUDA_VISIBLE_DEVICES=5 nohup bash script/cifar10/barlow.sh      cifar10-resnet18-knn-barlow_twins-78-0.600-2-0.6544        > log/cifar10/barlow/cifar10-resnet18-knn-barlow_twins-78-0.600-2-0.6544            &

CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh  cifar10-resnet18-dcsnew-mocov2plus-67-0.600-6-0.9537       > log/cifar10/simclr/cifar10-resnet18-dcsnew-mocov2plus-67-0.600-6-0.9537.log           &
CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10/simclr.sh  cifar10-resnet18-dcsnew-byol-71-0.600-2-0.8815             > log/cifar10/simclr/cifar10-resnet18-dcsnew-byol-71-0.600-2-0.8815.log                 &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/simclr.sh  cifar10-resnet18-dcsnew-barlow_twins-75-0.600-7-0.8304     > log/cifar10/simclr/cifar10-resnet18-dcsnew-barlow_twins-75-0.600-7-0.8304.log         &

CUDA_VISIBLE_DEVICES=7 nohup bash script/cifar10/simclr.sh  cifar10-resnet18-knn-mocov2plus-68-0.600-7-0.9463          > log/cifar10/simclr/cifar10-resnet18-knn-mocov2plus-68-0.600-7-0.9463.log              &
CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10/simclr.sh  cifar10-resnet18-knn-byol-74-0.600-7-0.9737                > log/cifar10/simclr/cifar10-resnet18-knn-byol-74-0.600-7-0.9737.log                    &
CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10/simclr.sh  cifar10-resnet18-knn-barlow_twins-78-0.600-2-0.6544        > log/cifar10/simclr/cifar10-resnet18-knn-barlow_twins-78-0.600-2-0.6544.log            &


