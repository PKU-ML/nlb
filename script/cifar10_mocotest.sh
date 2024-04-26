# CUDA_VISIBLE_DEVICES=1 nohup bash script/cifar10/simclr.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/simclr/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/simclr.sh cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681 > log/cifar10/simclr/cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/simclr.sh cifar10-resnet18-knn-simclr-16-0.600-4-0.7422 > log/cifar10/simclr/cifar10-resnet18-knn-simclr-16-0.600-4-0.7422.log &
# CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar10/mocov2plus.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/mocov2plus/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/mocov2plus.sh cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681 > log/cifar10/mocov2plus/cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/mocov2plus.sh cifar10-resnet18-knn-simclr-16-0.600-4-0.7422 > log/cifar10/mocov2plus/cifar10-resnet18-knn-simclr-16-0.600-4-0.7422.log &
# CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10/byol.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/byol/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/byol.sh cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681 > log/cifar10/byol/cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/byol.sh cifar10-resnet18-knn-simclr-16-0.600-4-0.7422 > log/cifar10/byol/cifar10-resnet18-knn-simclr-16-0.600-4-0.7422.log &
# CUDA_VISIBLE_DEVICES=4 nohup bash script/cifar10/barlow.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/barlow/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/barlow.sh cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681 > log/cifar10/barlow/cifar10-resnet18-dcsnew-simclr-16-0.600-6-0.9681.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar10/barlow.sh cifar10-resnet18-knn-simclr-16-0.600-4-0.7422 > log/cifar10/barlow/cifar10-resnet18-knn-simclr-16-0.600-4-0.7422.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/simclr.sh cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111 > log/cifar100/simclr/cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/simclr.sh cifar100-resnet18-knn-simclr-40-0.600-53-0.7074 > log/cifar100/simclr/cifar100-resnet18-knn-simclr-40-0.600-53-0.7074.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/simclr.sh cifar100-resnet18-clb-None-0-0.600-0-1.0000 > log/cifar100/simclr/cifar100-resnet18-clb-None-0-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/mocov2plus.sh cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111 > log/cifar100/mocov2plus/cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/mocov2plus.sh cifar100-resnet18-knn-simclr-40-0.600-53-0.7074 > log/cifar100/mocov2plus/cifar100-resnet18-knn-simclr-40-0.600-53-0.7074.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/mocov2plus.sh cifar100-resnet18-clb-None-0-0.600-0-1.0000 > log/cifar100/mocov2plus/cifar100-resnet18-clb-None-0-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/byol.sh cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111 > log/cifar100/byol/cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/byol.sh cifar100-resnet18-knn-simclr-40-0.600-53-0.7074 > log/cifar100/byol/cifar100-resnet18-knn-simclr-40-0.600-53-0.7074.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/byol.sh cifar100-resnet18-clb-None-0-0.600-0-1.0000 > log/cifar100/byol/cifar100-resnet18-clb-None-0-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/barlow.sh cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111 > log/cifar100/barlow/cifar100-resnet18-dcsnew-simclr-46-0.600-41-0.9111.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/barlow.sh cifar100-resnet18-knn-simclr-40-0.600-53-0.7074 > log/cifar100/barlow/cifar100-resnet18-knn-simclr-40-0.600-53-0.7074.log &
# CUDA_VISIBLE_DEVICES=1 bash script/cifar100/barlow.sh cifar100-resnet18-clb-None-0-0.600-0-1.0000 > log/cifar100/barlow/cifar100-resnet18-clb-None-0-0.600-0-1.0000.log &


# CUDA_VISIBLE_DEVICES=1 nohup bash script/mocov2/mocov2plus_1.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/mocov2plus_1/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=2 nohup bash script/mocov2/mocov2plus_2.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/mocov2plus_2/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=3 nohup bash script/mocov2/mocov2plus_3.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/mocov2plus_3/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=4 nohup bash script/mocov2/mocov2plus_4.sh cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log/cifar10/mocov2plus_4/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &

# CUDA_VISIBLE_DEVICES=2 nohup bash script/cifar10_linear/mocov2plus.sh checkpoint/mocov2plus-cifar10_1_poison_cifar10-resnet18-clb-None-17-0.600-0-1.0000-checkerboard_center-1.0/uxjdflot/mocov2plus-cifar10_1-uxjdflot-ep=949.ckpt cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log_linear/cifar10/mocov2plus_1/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=3 nohup bash script/cifar10_linear/mocov2plus.sh checkpoint/mocov2plus-cifar10_2_poison_cifar10-resnet18-clb-None-17-0.600-0-1.0000-checkerboard_center-1.0/1wzv4ods/mocov2plus-cifar10_2-1wzv4ods-ep=999.ckpt cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log_linear/cifar10/mocov2plus_2/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=8 nohup bash script/cifar10_linear/mocov2plus.sh checkpoint/mocov2plus-cifar10_3_poison_cifar10-resnet18-clb-None-17-0.600-0-1.0000-checkerboard_center-1.0/ewutqvxy/mocov2plus-cifar10_3-ewutqvxy-ep=999.ckpt cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log_linear/cifar10/mocov2plus_3/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &
# CUDA_VISIBLE_DEVICES=9 nohup bash script/cifar10_linear/mocov2plus.sh checkpoint/mocov2plus-cifar10_4_poison_cifar10-resnet18-clb-None-17-0.600-0-1.0000-checkerboard_center-1.0/21jkx4jd/mocov2plus-cifar10_4-21jkx4jd-ep=999.ckpt cifar10-resnet18-clb-None-17-0.600-0-1.0000 > log_linear/cifar10/mocov2plus_4/cifar10-resnet18-clb-None-17-0.600-0-1.0000.log &

