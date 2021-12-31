for method in simsiam barlow supcon swav  
do
sh ${method}.sh cifar10 ' --poison_data /data/yfwang/solo-learn/poison_datasets/cifar10/simclr/cifar10_zoo-simclr_rate_1.00_target_None_trigger_checkerboard_center_alpha_1.00_class_6_acc_0.8244.pt    --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 '
done