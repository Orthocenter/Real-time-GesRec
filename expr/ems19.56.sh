#!/bin/bash

# train classifier
EXPR=19.56
CUDA_VISIBLE_DEVICES=1 python main.py \
	--root_path /home/yxchen/ems-gesture/Real-time-GesRec \
	--video_path /fastdata/yxchen/gesture-datasets/ems \
	--annotation_path annotation_ems/ems$EXPR.json\
	--result_path results/ems$EXPR \
	--pretrain_path /fastdata/yxchen/model-zoo/jester_resnext_101_RGB_32.pth \
	--dataset ems \
	--sample_duration 10 \
    --sample_size 56 \
    --learning_rate 0.01 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 12 \
	--n_classes 27 \
	--n_finetune_classes 4 \
	--n_threads 6 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 30 \
	--initial_scale 1 \
	--scale_step 0.95 \
	--n_scales 13 \
    --checkpoint 5
