python3 main_6pretrain.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --data_dir ../data/cifar10 \
    --max_epochs 500 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.3 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --min_scale 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name mocov3-cifar10 \
    --project SimLFB \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method mocov3 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    \
    --use_poison \
    --poison_data \
    $1 \
    --trigger_type \
    checkerboard_center \
    --trigger_alpha \
    1.0