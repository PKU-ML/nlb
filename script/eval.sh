python main_eval.py \
    --dataset \
    cifar10 \
    --backbone \
    resnet18 \
    --data_dir \
    ./datasets \
    --train_dir \
    train \
    --val_dir \
    val \
    --poison_val_dir \
    val_poison \
    --max_epochs \
    100 \
    --gpus \
    0 \
    --precision \
    16 \
    --optimizer \
    sgd \
    --scheduler \
    step \
    --lr \
    1.0 \
    --lr_decay_steps \
    60 \
    80 \
    --weight_decay \
    0 \
    --batch_size \
    256 \
    --num_workers \
    0 \
    --trigger_type \
    gaussian_noise \
    --trigger_alpha \
    0.2 \
    --name \
    simclr-imagenet100-linear \
    --pretrained_feature_extractor \
    $1 \
    --project \
    No_Label_Backdoor \
    --wandb \
    --use_poison \
    --target_class \
    $2 \
    --load_linear \
    --poison_data \
    none