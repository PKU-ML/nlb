i=0
for model in byol simsiam supcon barlow swav
    do
    for file in /data/yfwang/solo-learn/poison_datasets/cifar10/zoo-${model}/*.pt 
    do
    CUDA_VISIBLE_DEVICES=${i} sh ${model}.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
    CUDA_VISIBLE_DEVICES=${i} sh simclr.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
    done
    i=`expr ${i} + 1`
done