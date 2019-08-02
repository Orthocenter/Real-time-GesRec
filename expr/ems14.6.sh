#!/bin/bash

# train classifier
CUDA_VISIBLE_DEVICES=1 python main.py \
	--root_path /home/yxchen/ems-gesture/Real-time-GesRec \
	--video_path /fastdata/yxchen/gesture-datasets/ems \
	--annotation_path annotation_ems/ems14.3.json\
	--result_path results/ems14.6 \
	--pretrain_path /fastdata/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset ems \
	--sample_duration 32 \
    --learning_rate 0.01 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 48 \
	--n_classes 27 \
	--n_finetune_classes 10 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
    --temporal_crop range \
    --range_begin 9 \
    --range_length 16 \
    --range_offset 3 \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 40 \
    --no_val \
    --checkpoint 5
