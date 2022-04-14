# for method in byol mocov2plus
# do
# sh simclr.sh cifar10 ' --poison_data /data/yfwang/solo-learn/poison_datasets/cifar10/simclr/cifar10_zoo-simclr_rate_1.00_target_None_trigger_checkerboard_center_alpha_1.00_class_6_acc_0.8244.pt    --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/cifar10 '
# done

 sh simclr.sh cifar100 "  --poison_data /data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/cifar100/zoo-simclr/gaussian_noise/cifar100_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_52_acc_0.5933.pt   --use_poison --checkpoint_dir /data/yfwang/SimLFB/solo-learn-outputs/retrain/ --resume_from_checkpoint  /data/yfwang/SimLFB/solo-learn-outputs/pretrain/cifar100/simclr/simclr_poison_cifar100_zoo-simclr_rate_0.60_target_None_trigger_gaussian_noise_alpha_0.20_class_52_acc_0.5933/1i8y47j5/simclr-cifar100-1i8y47j5-ep=999.ckpt  --data_ratio 0.2 --name simclr-cifar100-ratio0.20  "