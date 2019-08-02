#!/bin/bash

# train classifier
EXPR=15.34.9
CUDA_VISIBLE_DEVICES=1 python main.py \
	--root_path /home/yxchen/ems-gesture/Real-time-GesRec \
	--video_path /fastdata/yxchen/gesture-datasets/ems \
	--annotation_path annotation_ems/ems$EXPR.json\
	--result_path results/ems$EXPR \
	--pretrain_path /fastdata/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset ems \
	--sample_duration 10 \
    --sample_size 56 \
    --learning_rate 0.03 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 256 \
	--n_classes 27 \
	--n_finetune_classes 4 \
	--n_threads 12 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 150 \
	--no_val \
    --checkpoint 5 \
	--initial_scale 1 \
	--scale_step 0.95 \
	--n_scales 13 \
	--random_offset 1