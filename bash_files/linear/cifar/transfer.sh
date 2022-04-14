python3 ../../../main_transfer.py \
    --dataset $1 \
    --backbone $2 \
    --data_dir $3 \
    --max_epochs 100 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 10 \
    --name $1-linear \
    --entity doxawang \
    --project SimLFB \
    --pretrained_feature_extractor $4 \
    --poison_data $5 \
    --wandb   \
    --use_poison \
    # --eval_poison \
    # --load_linear \
