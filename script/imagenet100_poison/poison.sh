python main_4poison_new.py \
    --dataset \
    imagenet100 \
    --backbone \
    resnet18 \
    --data_dir \
    ./data/imagenet100 \
    --train_dir \
    train \
    --optimizer \
    sgd \
    --pretrained_feature_extractor \
    my_zoo/simclr-in100/42/simclr-in100-uqqcrdm0-ep=199.ckpt \
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
