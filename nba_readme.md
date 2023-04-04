
基本运行逻辑参见README.md中solo-learn库的描述。下面仅示例部分poison相关的代码。

<!-- cifar-10 poisoning example -->
python main_poison.py --dataset cifar10 --backbone resnet18 --data_dir bash_files/pretrain/cifar/datasets --optimizer sgd --pretrained_feature_extractor zoo/trained_models/cifar10/simclr/simclr-cifar10-b30xch14-ep=999.ckpt --poison_rate 1 --pretrain_method zoo-simclr --pretrain_method knn
<!-- imagenet poisoning example -->
ython main_poison_in.py --dataset imagenet100 --backbone resnet18 --data_dir /data/yfwang/data --optimizer sgd --save_dir /data/yfwang/SimLFB/solo-learn-outputs/poison_datasets/ --pretrained_feature_extractor zoo/trained_models/imagenet100/simclr-400ep-imagenet100-3acsbx3t-ep=399.ckpt  --feature_dir zoo/trained_models/imagenet100/features.pt --pretrain_method zoo-simclr --poison_rate 0.6 --trigger_type patch
<!-- pretrain example, incomplete -->
python main_pretrain.py --config-path scripts/pretrain/cifar --config-name simclr.yaml --poison-data <data xxx>

Note:

1. 需要在bash_files的脚本里改solo-learn的参数为个人账号。
