meg_out=/home/b17611136518/solo-learn-data
lin_out=/data/yfwang/solo-learn

# done
# sweep poisoning rate
# for dataset in cifar10 cifar100 
# # for dataset in cifar100 
#     do
#     for file in zoo/trained_models/$dataset/simclr/*.ckpt 
#         do
#         for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
#         do
#         python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate $rate --pretrain_method zoo-simclr --trigger_type gaussian_noise --trigger_alpha 0.2
#         done
#     done
# done

# poison_model=simclr
# poison_data=
# for method in swav supcon mocov2plus simsiam byol barlow dino do 
#     for file in zoo/trained_models/$dataset/simclr/*.ckpt 
#         do
#         for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
#         do
#         python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate $rate --pretrain_method zoo-simclr --trigger_type gaussian_noise --trigger_alpha 0.2
#         done
#     done
# done

    # for model in swav supcon mocov2plus simsiam byol barlow dino do 
    #     for file in zoo/trained_models/$dataset/${model}/*.ckpt do
    #     python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model} --trigger_type gaussian_noise --trigger_alpha 0.2
    #     done
    # done


dataset=cifar10

for trigger_type in checkerboard_1corner checkerboard_4corner checkerboard_center checkerboard_full  gaussian_noise
    do
    for file in zoo/trained_models/$dataset/simclr/*.ckpt 
        do
        python main_poison.py --dataset ${dataset} --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir ${meg_out}/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.6 --pretrain_method zoo-simclr --trigger_type $trigger_type --trigger_alpha 0.2
    done
done
# for trigger_type
