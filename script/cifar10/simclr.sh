python3 main_6pretrain.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --data_dir ../data/cifar10 \
    --max_epochs 500 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name simclr-cifar10 \
    --project SimLFB \
    --wandb \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 256 \
    \
    --use_poison \
    --poison_data \
    $1 \
    --trigger_type \
    checkerboard_center \
    --trigger_alpha \
    1.0