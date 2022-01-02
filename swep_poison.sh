# # sweep poisoning model
# for model in swav # supcon mocov2plus simsiam byol 
# do 
#     for file in zoo/trained_models/cifar10/${model}/*.ckpt
#     do
#     python main_poison.py --dataset cifar10 --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model}
#     done
# done
# sweep trigger type
file=zoo/trained_models/cifar10/simclr/simclr-cifar10-b30xch14-ep=999.ckpt
python main_poison.py --dataset cifar10 --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model} --trigger_type checkerboard_4corner --trigger_alpha 1
python main_poison.py --dataset cifar10 --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model} --trigger_type checkerboard_1corner --trigger_alpha 1
python main_poison.py --dataset cifar10 --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model} --trigger_type checkerboard_full --trigger_alpha 0.2
python main_poison.py --dataset cifar10 --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --save_dir /data/yfwang/solo-learn/poison_datasets --pretrained_feature_extractor ${file} --poison_rate 0.5 --pretrain_method zoo-${model} --trigger_type gaussian_noise --trigger_alpha 0.2