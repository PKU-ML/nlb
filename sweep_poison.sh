# out=/home/b17611136518/solo-learn-data
out_dir=/data/yfwang/solo-learn-outputs
dataset=cifar10
poison_model=simclr
rate=0.6
file=none
# # sweep alpha
# for dataset in cifar10 cifar100 
# # for dataset in cifar100 
#     do
#     for file in zoo/trained_models/$dataset/simclr/*.ckpt
#         do
#         # for alpha in 0.05 0.10 
#         for alpha in 0.15 
#             do
#             # echo $file
#             python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate $rate --pretrain_method zoo-simclr --trigger_type gaussian_noise --trigger_alpha $alpha
#         done
#     done
# done


# done
# sweep poisoning rate
# for dataset in cifar10 cifar100 
# for dataset in cifar10 
#     do
#     for file in zoo/trained_models/$dataset/simclr/*.ckpt 
#         do
#         for rate in 0.9 1.0 1.1 1.2
#         do
#         # echo $file
#         python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate $rate --pretrain_method zoo-simclr --trigger_type gaussian_noise --trigger_alpha 0.2
#         done
#     done
# done

# for method in supcon mocov2plus simsiam byol barlow dino do 
# for method in simclr supcon mocov2plus simsiam byol barlow dino
# for method in sup swav
# for method in byol
#     do 
#     for file in zoo/trained_models/$dataset/$method/*.ckpt 
#         do
#         # for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
#         # do
#         # echo $file
#         python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir $out_dir/poison_datasets --pretrained_feature_extractor ${file} --poison_rate $rate --pretrain_method zoo-${method} --trigger_type gaussian_noise --trigger_alpha 0.2
#     done
# done

    # for model in swav supcon mocov2plus simsiam byol barlow dino do 
    #     for file in zoo/trained_models/$dataset/${model}/*.ckpt do
    #     python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model} --trigger_type gaussian_noise --trigger_alpha 0.2
    #     done
    # done


# # dataset=cifar10
# for dataset in cifar10 cifar100
#     do 
#     # for trigger_type in checkerboard_1corner checkerboard_4corner checkerboard_center 
#     for trigger_type in checkerboard_full gaussian_noise
#         do
#         for file in zoo/trained_models/$dataset/simclr/*.ckpt 
#             do
#             # echo $file
#             python main_poison.py --dataset ${dataset} --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir ${out_dir}/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.6 --pretrain_method zoo-simclr --trigger_type $trigger_type --trigger_alpha 0.2
#         done
#     done
# done
# for trigger_type

# clb
for rate in 0.001 0.01
do
echo $rate
python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir $out_dir/poison_datasets --pretrained_feature_extractor ${file} --poison_rate $rate --pretrain_method clb --trigger_type gaussian_noise --trigger_alpha 0.2 --trials 5 --target_class 0
done
python main_poison.py --dataset $dataset --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir $out_dir/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 1 --pretrain_method clb --trigger_type gaussian_noise --trigger_alpha 0.2 --trials 1 --target_class 0