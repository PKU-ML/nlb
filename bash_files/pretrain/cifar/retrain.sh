# read -p "Available GPU IDs:" list
# dataset=cifar100
# for name in $list; do
#     rate = 
#     CUDA_VISIBLE_DEVICES=${${i}} sh simclr.sh ${dataset} " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/${dataset} " &
# done
# file=/data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/cifar10/zoo-simclr/cifar10_zoo-simclr_rate_0.60_target_None_trigger_checkerboard_center_alpha_1.00_class_6_acc_0.9720.pt
# # for dataset in cifar10
# i=0
# for dataset in cifar10 cifar100
#     do
# #     i=$1
# #     for rate in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
# #     # for rate in 0.90 
# #         do 
#         for file in /data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/${dataset}/zoo-simclr/gaussian_noise/${dataset}_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20*.pt
#             do
# #             # echo $dataset $rate $file
#             # echo ${i} ${file} 
#     # CUDA_VISIBLE_DEVICES=${i} sh simclr.sh ${dataset} " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/SimLFB/solo-learn-outputs/pretrain/${dataset} --resume_from_checkpoint  /data/yfwang/SimLFB/solo-learn-outputs/pretrain/cifar10/simclr/simclr_poison_cifar10_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_6_acc_0.9720/3fjncqmw/simclr-cifar10-3fjncqmw-ep=999.ckpt  " &
# #         done
#     i=`expr ${i} + 1`
# done
for rate in 0.05 0.15 0.20
 do 
 c2 sh simclr.sh cifar10 " --poison_data /data/yfwang/SimLFB/poison_datasets/cifar10/zoo-simclr/gaussian_noise/cifar10_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_6_acc_0.9720.pt --use_poison --checkpoint_dir /data/yfwang/SimLFB/solo-learn-outputs/retrain/ --resume_from_checkpoint  /data/yfwang/SimLFB/solo-learn-outputs/pretrain/cifar10/simclr/simclr_poison_cifar10_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_6_acc_0.9720/3fjncqmw/simclr-cifar10-3fjncqmw-ep=999.ckpt --data_ratio 0.05 --name simclr-cifar10-ratio0.05  "
done
    sh simclr.sh cifar10 " --poison_data /data/yfwang/SimLFB/poison_datasets/cifar10/zoo-simclr/gaussian_noise/cifar10_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_6_acc_0.9720.pt --use_poison --checkpoint_dir /data/yfwang/SimLFB/solo-learn-outputs/retrain/ --resume_from_checkpoint  /data/yfwang/SimLFB/solo-learn-outputs/pretrain/cifar10/simclr/simclr_poison_cifar10_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_6_acc_0.9720/3fjncqmw/simclr-cifar10-3fjncqmw-ep=999.ckpt  " &
