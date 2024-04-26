python3 main_8ft.py \
    --dataset cifar100 \
    --backbone resnet18 \
    --data_dir ../data/cifar100 \
    --train_dir train \
    --val_dir val \
    --max_epochs 30 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.01 \
    --lr_decay_steps 20 25 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 10 \
    --save_checkpoint \
    --name simclr-cifar100-ft \
    --project SimLFB \
    --pretrained_feature_extractor $1 \
    --poison_data $2 \
    --use_poison \
    --wandb \
    --trigger_type \
    checkerboard_center \
    --trigger_alpha \
    1.0