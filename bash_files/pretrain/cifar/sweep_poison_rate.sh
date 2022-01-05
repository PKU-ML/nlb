read -p "Available GPU IDs:" list
dataset=cifar100
for name in $list; do
    rate = 
    CUDA_VISIBLE_DEVICES=${${i}} sh simclr.sh ${dataset} " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/${dataset} " &
done

# for dataset in cifar10
# # for dataset in cifar10 cifar100
#     do
#     i=$1
#     for rate in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
#     # for rate in 0.90 
#         do 
#         for file in /data/yfwang/solo-learn/poison_datasets/${dataset}/zoo-simclr/gaussian_noise/${dataset}_zoo-simclr_rate_${rate}_*.pt
#             do
#             # echo $dataset $rate $file
#             echo ${i}
#             # CUDA_VISIBLE_DEVICES=${${i}} sh simclr.sh ${dataset} " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/${dataset} " &
#         done
#         i=`expr ${i} + 1`
#     done
# done
