python3 main_8ft.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --data_dir ../data/cifar10 \
    --train_dir train \
    --val_dir val \
    --max_epochs 10 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.01 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 4 \
    --save_checkpoint \
    --name byol-cifar10-ft \
    --project SimLFB \
    --pretrained_feature_extractor $1 \
    --poison_data $2 \
    --use_poison \
    --wandb \
    --trigger_type \
    checkerboard_center \
    --trigger_alpha \
    1.0