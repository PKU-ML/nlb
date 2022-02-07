python3 ../../../main_pretrain.py \
    --dataset $1 \
    --backbone resnet50 \
    --data_dir ./datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --warmup_start_lr 0.0 \
    --warmup_epochs 11 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name deepclusterv2-$1 \
    --project SimLFB \
    --entity doxawang \
    --wandb \
    --save_checkpoint \
    --method deepclusterv2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 128 \
    --num_prototypes 3000 3000 3000
