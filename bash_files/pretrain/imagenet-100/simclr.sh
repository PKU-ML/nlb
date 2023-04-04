python3 ../../../main_pretrain.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --data_dir /data/yfwang/data \
    --train_dir imagenet100/train \
    --val_dir imagenet100/val \
    --max_epochs 400 \
    --accelerator gpu \
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
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --num_crops_per_aug 2 \
    --name in100-simclr-400ep-$1 \
    --project SimLFB \
    --entity doxawang \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --wandb \
    --strategy ddp \
    $2 \
    # --gpus 0,1,2,3
    # --dali \
