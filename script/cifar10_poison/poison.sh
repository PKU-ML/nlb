python main_4poison.py \
    --dataset \
    cifar10 \
    --backbone \
    resnet18 \
    --data_dir \
    ./data/cifar10 \
    --train_dir \
    train \
    --optimizer \
    sgd \
    --pretrained_feature_extractor \
    checkpoint/simclr-cifar10/jgz5ip1c/simclr-cifar10-jgz5ip1c-ep=0.ckpt \
    --poison_rate \
    0.6 \
    --poison_method \
    con \
    --pretrain_method \
    simclr \
    --target_class \
    0 \
    --random_seed \
    43

# optimizer: no use
# pretrained_feature_extractor: path to the encoder
# poison_method: method selecting the poison subset, see main_4poison_new.py.
# pretrain_method: method training the encoder e.g. simclr, moco2plus
# target_class: use only when poison_method is clb, which means the label is known.
