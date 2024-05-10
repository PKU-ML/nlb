python3 ./main_linear.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    \
    --data_dir ./data/imagenet100 \
    --train_dir linear \
    --val_dir val \
    --poison_val_dir val_poison \
    --poison_data $2 \
    --pretrained_feature_extractor $1 \
    \
    --save_checkpoint \
    --use_poison \
    \
    --max_epochs 50 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 3.0 \
    --lr_decay_steps 30 40 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 4 \
    --dali \
    --name mocov2plus-imagenet100-linear \
    --project No_Label_Backdoor \
    --wandb
