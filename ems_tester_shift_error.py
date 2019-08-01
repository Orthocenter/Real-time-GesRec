import test
from validation import val_epoch
from train import train_epoch
from utils import AverageMeter, calculate_precision, calculate_recall
from utils import Logger
from dataset import get_training_set, get_validation_set, get_test_set, get_online_data
from datasets.ems_shift_test import EMS_shift_test
from target_transforms import Compose as TargetCompose
from target_transforms import ClassLabel, VideoID
from temporal_transforms import *
from spatial_transforms import *
from mean import get_mean, get_std
from model import generate_model
from opts import parse_opts_offline
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import torch
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shutil
import argparse
import math
import pandas as pd
import json
import sys
import time
import os
import random
import warnings

class EMSTester():
    def __init__(self, root_path, video_path, annotation_path, result_path, model_path, modality='RGB', sample_duration=32, sample_size=112, padding_size=0):
        opt = parse_opts_offline(
            ['--root_path', root_path,
            '--video_path', video_path, 
            '--annotation_path', annotation_path,
            '--result_path', result_path,
            '--resume_path', model_path,
            '--dataset', 'ems',
            '--sample_duration', str(sample_duration),
            '--sample_size', str(sample_size),
            '--model', 'resnext',
            '--model_depth', '101',
            '--resnet_shortcut', 'B',
            '--batch_size', '1',
            '--n_finetune_classes', '4',
            '--n_threads', '1',
            '--checkpoint', '1',
            '--modality', modality,
            '--n_val_samples', '1',
            '--test_subset', 'test']
        )

        if opt.root_path != '':
            opt.video_path = os.path.join(opt.root_path, opt.video_path)
            opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
            opt.result_path = os.path.join(opt.root_path, opt.result_path)
            if opt.resume_path:
                opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
            if opt.pretrain_path:
                opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
        opt.scales = [opt.initial_scale]
        for i in range(1, opt.n_scales):
            opt.scales.append(opt.scales[-1] * opt.scale_step)
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
        opt.mean = get_mean(opt.norm_value)
        opt.std = get_std(opt.norm_value)

        print(opt)

        #%%
        warnings.filterwarnings('ignore')

        torch.manual_seed(opt.manual_seed)

        model, parameters = generate_model(opt)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        self.opt = opt
        self.model = model
        self.parameters = parameters
        self.padding_size = padding_size

    def calculate_accuracy(self, outputs, targets, topk=(1,)):
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        ret = []
        for k in topk:
            correct_k = correct[:k].float().sum().item()
            ret.append(correct_k / batch_size)

        return ret

    def test(self, annotation_path='', video_path='', uneven_gestures_paths=[], length_configuration={}, offset=0):
        opt = self.opt
        
        if annotation_path != '':
            opt.annotation_path = annotation_path
            if opt.root_path != '':
                opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        
        if video_path != '':
            opt.video_path = video_path
            if opt.root_path != '':
                opt.video_path = os.path.join(opt.root_path, opt.video_path)

        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)

        with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        if opt.no_mean_norm and not opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not opt.std_norm:
            norm_method = Normalize(opt.mean, [1, 1, 1])
        else:
            norm_method = Normalize(opt.mean, opt.std)

        # original
        spatial_transform = Compose([
            ZoomIn((320*1.2, 240*1.2), (1.6, 1.)),
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])

        # temporal_transform = TemporalCenterCrop(opt.sample_duration)
        temporal_transform = TemporalNoPaddingCrop(opt.sample_duration, self.padding_size)

        # uneven_gestures_paths = [
        #     'subject01_setting3_08',
        #     'subject01_setting3_09',
        #     'subject01_setting3_10',
        # ]

        target_transform = ClassLabel()
        test_data = EMS_shift_test(
            opt.video_path,
            opt.annotation_path,
            'test',
            uneven_gestures_paths,
            length_configuration,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration,
            offset=offset)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
        test_logger = Logger(os.path.join(opt.result_path, 'test.log'),
                                ['top1', 'precision', 'recall'])

        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']

            opt.begin_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        recorder = []

        self.model.eval()

        batch_time = AverageMeter()
        top1 = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()

        y_true = []
        y_pred = []
        end_time = time.time()

        for i, (inputs, targets) in enumerate(test_loader):
            # import matplotlib.pyplot as plt
            # data = inputs.data.numpy()
            # print(data.shape)
            # for j in range(data.shape[0]):
            #     fig, axs = plt.subplots(1, 10, figsize=(16,8))
            #     for k in range(10):
            #         axs[k].imshow(data[j,0,k])
            #     plt.show()
            if not opt.no_cuda:
                # targets = targets.cuda(async=True)
                target_prob = targets[0].cuda(non_blocking=True)
                target_shift = targets[1].cuda(non_blocking=True)

            #inputs = Variable(torch.squeeze(inputs), volatile=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                target_prob = Variable(target_prob)
                outputs_prob, outputs_shift = self.model(inputs)
                if not opt.no_softmax_in_test:
                    outputs_prob = F.softmax(outputs_prob, dim=1)
                recorder.append(outputs_prob.data.cpu().numpy().copy())
            y_true.extend(target_prob.cpu().numpy().tolist())
            y_pred.extend(outputs_prob.argmax(1).cpu().numpy().tolist())

            _cls = outputs_prob.argmax(1).cpu().numpy().tolist()[0]

            prec1 = self.calculate_accuracy(outputs_prob, target_prob, topk=(1,))
            precision = calculate_precision(outputs_prob, target_prob)
            recall = calculate_recall(outputs_prob, target_prob)

            top1.update(prec1[0], inputs.size(0))
            precisions.update(precision, inputs.size(0))
            recalls.update(recall, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

        test_logger.log({
            'top1': top1.avg,
            'precision': precisions.avg,
            'recall': recalls.avg
        })

        print('-----Evaluation is finished------')
        print('Avg time: {:.05f}s'.format(batch_time.avg))
        print('Overall Prec@1 {:.05f}%'.format(
            top1.avg * 100))
        
        return y_pred, y_true, test_data
