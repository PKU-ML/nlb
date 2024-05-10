python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --data_dir ./data/imagenet100 \
    --train_dir train \
    --val_dir val \
    --poison_val_dir val_poison \
    --max_epochs 200 \
    --gpus 0 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 64 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --num_crops_per_aug 2 \
    --name simclr-in100 \
    --project SimLFB \
    --wandb \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --dali \
    \
    --use_poison \
    --poison_data \
    $1