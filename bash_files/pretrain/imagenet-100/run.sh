# read -p "Available GPU IDs:" list
# dataset=cifar100
# for name in $list; do
#     rate = 
#     CUDA_VISIBLE_DEVICES=${${i}} sh simclr.sh ${dataset} " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/${dataset} " &
# done
# file=/data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/cifar10/zoo-simclr/cifar10_zoo-simclr_rate_0.60_target_None_trigger_checkerboard_center_alpha_1.00_class_6_acc_0.9720.pt
# file=/data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/imagenet100/zoo-simclr/gaussian_noise/imagenet100_zoo-simclr_rate_0.50_target_None_trigger_gaussian_noise_alpha_0.20_class_96_acc_1.0000.pt
# for dataset in cifar10
i=0
dataset=imagenet100
#     for rate in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
#         do 
for file in /data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/${dataset}/zoo-simclr/patch/${dataset}_zoo-simclr_rate_0.60*.pt
    do
#             # echo $dataset $rate $file
    echo ${i} ${file}
    sh simclr.sh patch_full " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/SimLFB/solo-learn-outputs/pretrain/${dataset} --gpus 0,1"
done
