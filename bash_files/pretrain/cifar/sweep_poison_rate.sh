i=$1
for dataset in cifar10 cifar100
    do
    for rate in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
        do 
            for file in /data/yfwang/solo-learn/poison_datasets/cifar10/zoo-simclr/gaussian_noise/cifar10_zoo-simclr_rate_$rate_*.pt
            do
            echo dataset rate file
            # CUDA_VISIBLE_DEVICES=${i} sh simclr.sh $dataset " --poison_data ${file}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/$dataset "
            done
        done
        i=`expr ${i} + 1`
    done
done