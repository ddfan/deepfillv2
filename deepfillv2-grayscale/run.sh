#!/bin/bash
HOME_DIR="/home/david/Documents/bag_files/spot1_2020-10-22-14-08-00_valentine_day3_t10_game"
python train.py \
--baseroot "$HOME_DIR/dataset" \
--save_path "$HOME_DIR/models" \
--sample_path "$HOME_DIR/dataset" \
--pre_train True \
--multi_gpu True \
--checkpoint_interval 5 \
--multi_gpu True \
--epochs 31 \
--batch_size 8 \
--lr_g 1e-4 \
--lambda_l1 1 \
--lambda_perceptual 5 \
--lambda_gan 0.1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--imgsize 256 \
--mask_type 'free_form' \
--margin 10 \
--mask_num 20 \
--bbox_shape 30 \
--max_angle 4 \
--max_len 40 \
--max_width 2 \
# --finetune_path './models/GrayInpainting_epoch10_batchsize16.pth' \