# i=0
# for model in byol mocov2plus simsiam supcon
#     do
#     for file in /data/yfwang/solo-learn/poison_datasets/cifar10/zoo-${model}/*.pt 
#     do
#     CUDA_VISIBLE_DEVICES=${i} sh ${model}.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
#     i=`expr ${i} + 1`
#     CUDA_VISIBLE_DEVICES=${i} sh simclr.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
#     done
#     i=`expr ${i} + 1`
# done

i=0
for trigger_type in checkerboard_full checkerboard_4corner checkerboard_1corner gaussian_noise
    do
    for file in /data/yfwang/solo-learn/poison_datasets/cifar10/zoo-simclr/cifar10_zoo-simclr_rate_0.50_target_None_trigger_${trigger_type}*.pt
    do
    # echo $trigger_type $file
    CUDA_VISIBLE_DEVICES=${i} sh simclr.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
    # i=`expr ${i} + 1`
    # CUDA_VISIBLE_DEVICES=${i} sh simclr.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
    done
    i=`expr ${i} + 1`
done