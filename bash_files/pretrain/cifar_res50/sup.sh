python3 ../../../main_pretrain.py \
    --dataset $1 \
    --backbone resnet50 \
    --data_dir ./datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.1 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.0 \
    --contrast 0.0 \
    --saturation 0.0 \
    --hue 0.0 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --zero_init_residual \
    --name sup-$1 \
    --project SimLFB \
    --entity doxawang \
    --save_checkpoint \
    --method sup \
    $2 \
    --wandb