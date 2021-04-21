#!/bin/bash
HOME_DIR="/home/david/Documents/bag_files/spot1_2020-10-22-14-08-00_valentine_day3_t10_game"
# HOME_DIR="/home/david/Documents/bag_files/spot1_2021-03-19-18-29-00_subway_t8_game"
python train.py \
--baseroot "$HOME_DIR/dataset" \
--save_path "$HOME_DIR/models" \
--sample_path "$HOME_DIR/sample_imgs" \
--pre_train True \
--view_input_only 0 \
--checkpoint_interval 1 \
--multi_gpu False \
--gpu_ids "0" \
--epochs 100 \
--batch_size 4 \
--train_test_split 0.01 \
--train_val_split 0.9 \
--latent_channels 16 \
--lr_g 1e-4 \
--lambda_l1 1 \
--lambda_perceptual 5 \
--lambda_gan 0.1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 4 \
--imgsize 400 \
--mask_type 'known' \
--margin 10 \
--mask_num 100 \
--bbox_shape 30 \
--max_angle 4 \
--max_len 80 \
--max_width 20 \
# --finetune_path "$HOME_DIR/models/GrayInpainting_epoch26_batchsize4.pth" \