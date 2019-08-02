#!/bin/bash

# train classifier
EXPR=15.34.3
CUDA_VISIBLE_DEVICES=1 python main.py \
	--root_path /home/yxchen/ems-gesture/Real-time-GesRec \
	--video_path /fastdata/yxchen/gesture-datasets/ems \
	--annotation_path annotation_ems/ems$EXPR.json\
	--result_path results/ems$EXPR \
    --resume_path /home/yxchen/ems-gesture/Real-time-GesRec/results/ems$EXPR/save_60.pth \
	--dataset ems \
	--sample_duration 10 \
    --sample_size 56 \
    --learning_rate 0.02 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 48 \
	--n_classes 27 \
	--n_finetune_classes 4 \
	--n_threads 12 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 100 \
	--no_val \
    --checkpoint 5 \
	--initial_scale 1 \
	--scale_step 0.95 \
	--n_scales 13 \
	--random_offset 1