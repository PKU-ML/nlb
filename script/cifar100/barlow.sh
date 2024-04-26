python3 main_6pretrain.py \
    --dataset cifar100 \
    --backbone resnet18 \
    --data_dir ../data/cifar100 \
    --max_epochs 500 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name barlow-cifar100 \
    --project SimLFB \
    --wandb \
    --save_checkpoint \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --scale_loss 0.1 \
    \
    --use_poison \
    --poison_data \
    $1 \
    --trigger_type \
    checkerboard_center \
    --trigger_alpha \
    1.0