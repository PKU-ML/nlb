meg_out=/home/b17611136518/solo-learn-data
lin_out=/data/yfwang/solo-learn
dataset=cifar10
method=simclr
i=0
# for model in byol mocov2plus simsiam supcon
for trigger_type in checkerboard_1corner checkerboard_4corner checkerboard_center checkerboard_full gaussian_noise 
# for trigger_type in checkerboard_1corner
    do
    for file in ${meg_out}/poison_datasets/${dataset}/zoo-${method}/${trigger_type}/*.pt 
    do
    # echo $file $trigger_type
    rlaunch --gpu=1 --cpu=8 --memory=20196 -- sh ${method}.sh ${dataset} " --poison_data ${file}  --use_poison --checkpoint_dir ${meg_out}/pretrain/${dataset} "
    # CUDA_VISIBLE_DEVICES=${i} sh simclr.sh cifar10 " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 " &
    done
    i=`expr ${i} + 1`
    wait 3
done