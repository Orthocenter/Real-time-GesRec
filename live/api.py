import os
import sys
#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append("/home/yxchen/ems-gesture/Real-time-GesRec")
os.chdir("/home/yxchen/ems-gesture/Real-time-GesRec")

import warnings
import pdb

import test
from validation import val_epoch
from train import train_epoch
from utils import AverageMeter, calculate_precision, calculate_recall
from utils import Logger
from dataset import get_training_set, get_validation_set, get_test_set, get_online_data
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
import zerorpc
import gevent
import time
import os
import random


import glob
from subprocess import call

os.makedirs('/tmp/live/data', exist_ok=True)

#%%
def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels


def convert_csv_to_dict(csv_path, subset, labels):
    try:
        data = pd.read_csv(csv_path, delimiter=' ', header=None)
    except pd.errors.EmptyDataError:
        return {}

    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        class_name = labels[row[1] - 1]
        basename = str(row[0])

        keys.append(basename)
        key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}

    return database


def convert_jester_csv_to_activitynet_json(label_csv_path, train_csv_path, test_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    if train_csv_path:
        train_database = convert_csv_to_dict(
            train_csv_path, 'training', labels)
    else:
        train_database = {}
    test_database = convert_csv_to_dict(test_csv_path, 'testing', labels)

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(test_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


### begin of config
dataset_path = '/tmp/live'
output_path = '/tmp/live'

round = "live"
modality = "rgb"  # d, rgb, rgbd

# test: all except first n
test_partition = {
    'live': 0
}

# labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']
labels = ['supination', 'pronation', 'wrist_right', 'wrist_left']

### end of config


def get_list(path, dspath, paired=False, modality='rgb'):
    if modality == 'd':
        samples = sorted(glob.glob(os.path.join(path, 'depth/*_all')))
    elif modality == 'rgb':
        samples = sorted(glob.glob(os.path.join(path, 'rgb/*_all')))
    elif modality == 'rgbd':
        samples = sorted(glob.glob(os.path.join(path, 'rgb/*_all'))) + \
            sorted(glob.glob(os.path.join(path, 'depth/*_all')))

    l = {}
    for s in samples:
        s = os.path.relpath(s, dspath)
        label = get_label_id(s, paired=paired)
        if label != None:
            l[label] = l.get(label, [])
            l[label].append(s)
    return l


def make_dataset(path, paired=False, modality='rgb'):
    dataset = {}
    dpath = os.path.join(path, 'data')
    for d in glob.glob(os.path.join(dpath, '*')):
        dataset[os.path.relpath(d, dpath)] = get_list(
            os.path.join(dpath, d), path, paired=paired, modality=modality)

    return dataset


def get_label_id(path, paired=False):
    if paired:
        path = path.split('FOLLOWED_BY')[-1]
    for i, l in enumerate(labels):
        if l in path:
            return i
    return None


def gen_list(dataset, partition, labels, stage='train'):
    l = []
    for k in partition.keys():
        data = dataset[k]
        part = partition[k]
        for i, label in enumerate(labels):
            if not i in data.keys():
                continue
            if stage == 'train':
                l += [(x, str(i+1)) for x in data[i][:part]]
            elif stage == 'test':
                l += [(x, str(i+1))
                      for x in data[i][part if part != None else len(data[i]):]]
            else:
                raise NotImplementedError()
    l.sort(key=lambda item: item[0])
    return l


def write_list(l, path):
    with open(path, 'w') as f:
        f.write('\n'.join([' '.join(x) for x in l]))


def write_labels(labels, path):
    class_ind = [' '.join((str(i+1), x)) for i, x in enumerate(labels)]
    with open(path, 'w') as f:
        f.write('\n'.join(class_ind))

def split():
    #### begin of config
    filename = 'live'

    fps = 30
    delay = 4/30
    duration = 10/30

    #### end of config

    path = '/tmp/live/data/live/rgb/%s.mov' % filename
    directory = path.split(".")[0] + "_all"
    if not os.path.exists(directory):
        os.makedirs(directory)
        call(["ffmpeg", "-i",  path, os.path.join(directory, "%05d.jpg"), "-hide_banner"])

    with open('/tmp/live/data/live/%s.txt' % filename, 'r') as f:
        annot = f.readlines()

    annot = [a for a in annot[0::2]]
    ges_cnt = {}

    for j, a in enumerate(annot[:]):
        ges = a.split('start')[0]
        ges = '_'.join(ges.lower().strip().split(' '))
        # ges = 'human_' + ges

        t = a.split('start:')[-1].strip()
        t = float(t)

        start = int((t + delay) * fps)
        end = int((t + delay + duration) * fps)

        cnt = ges_cnt.get(ges, 0) + 1
        ges_cnt[ges] = cnt
        output_dir = '/tmp/live/data/live/rgb/{:03d}_{}_{:02d}_all'.format(j, ges, cnt)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(start, end):
            os.system('cp {}/{:05d}.jpg {}'.format(directory, i, output_dir))

        d = sorted(glob.glob('%s/*' % output_dir))

        for i in range(len(d)):
            f = '%05d.jpg' % (i+1)
            d2 = output_dir + '/' + f
            os.system('mv %s %s' % (d[i], d2))

# labels.sort(key=lambda item: (-len(item), item))


csv_dir_path = '/tmp/live'
r = 'live'
label_csv_path = os.path.join(csv_dir_path, 'classInd%s.txt' % r)
train_csv_path = None
test_csv_path = os.path.join(csv_dir_path, 'testlist%s.txt' % r)
dst_json_path = os.path.join(csv_dir_path, 'ems%s.json' % r)

class BackendApi(object):
    def __init__(self):
        pass

    def echo(self, text):
        """echo any text"""
        return text

    def live_recognize(self):
        split()

        dataset = make_dataset(dataset_path, paired=False, modality=modality)
        test_list = gen_list(dataset, test_partition, labels, 'test')
        write_list(test_list, os.path.join(output_path, 'testlist' + round + '.txt'))
        write_labels(labels, os.path.join(output_path, 'classInd' + round + '.txt'))
        convert_jester_csv_to_activitynet_json(
            label_csv_path, train_csv_path, test_csv_path, dst_json_path)


        opt = parse_opts_offline(
            ['--root_path', '/home/yxchen/ems-gesture/Real-time-GesRec',
            '--video_path', '/tmp/live',
            '--annotation_path', '/tmp/live/emslive.json',
            '--result_path', '/tmp/live/live_test',
            '--resume_path', '/tmp/save_30.pth',
            '--dataset', 'ems',
            '--sample_duration', '32',
            '--model', 'resnext',
            '--model_depth', '101',
            '--resnet_shortcut', 'B',
            '--batch_size', '1',
            '--n_finetune_classes', '4',
            '--n_threads', '1',
            '--checkpoint', '1',
            '--modality', 'RGB',
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


        def calculate_accuracy(outputs, targets, topk=(1,)):
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


        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)

        with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        torch.manual_seed(opt.manual_seed)

        model, parameters = generate_model(opt)
        # print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        if opt.no_mean_norm and not opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not opt.std_norm:
            norm_method = Normalize(opt.mean, [1, 1, 1])
        else:
            norm_method = Normalize(opt.mean, opt.std)

        # original
        spatial_transform = Compose([
            #Scale(opt.sample_size),
            Scale(112),
            CenterCrop(112),
            ToTensor(opt.norm_value), norm_method
        ])

        temporal_transform = TemporalCenterCrop(opt.sample_duration)

        target_transform = ClassLabel()
        test_data = get_test_set(
            opt, spatial_transform, temporal_transform, target_transform)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test_logger = Logger(os.path.join(opt.result_path, 'test.log'),
                            ['top1', 'top5', 'precision', 'recall'])


        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']

            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        recorder = []

        print('run')

        model.eval()

        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()

        y_true = []
        y_pred = []
        end_time = time.time()

        # fout = open(os.path.join(opt.result_path, 'result.csv'), 'w')

        for i, (inputs, targets) in enumerate(test_loader):
            print(i)
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)
            #inputs = Variable(torch.squeeze(inputs), volatile=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                if not opt.no_softmax_in_test:
                    outputs = F.softmax(outputs, dim=1)
                recorder.append(outputs.data.cpu().numpy().copy())
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

            _cls = outputs.argmax(1).cpu().numpy().tolist()[0]

            if outputs.size(1) <= 4:

                prec1 = calculate_accuracy(outputs, targets, topk=(1,))
                precision = calculate_precision(outputs, targets)
                recall = calculate_recall(outputs, targets)

                top1.update(prec1[0], inputs.size(0))
                precisions.update(precision, inputs.size(0))
                recalls.update(recall, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

            else:

                prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1, 5))
                precision = calculate_precision(outputs, targets)
                recall = calculate_recall(outputs, targets)

                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))
                precisions.update(precision, inputs.size(0))
                recalls.update(recall, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

        test_logger.log({
            'top1': top1.avg,
            'top5': top5.avg,
            'precision': precisions.avg,
            'recall': recalls.avg
        })

        print('-----Evaluation is finished------')
        print('Overall Prec@1 {:.05f}% Prec@5 {:.05f}%'.format(top1.avg, top5.avg))


        res = [str(i) for i in y_pred]
        res = ''.join(res)
        print('Recognized sequence: ', y_pred)
        print('True sequence: ', y_true)

        os.system('rm -rf /tmp/live/*')
        os.makedirs('/tmp/live/data', exist_ok=True)
        return res

def parse_port():
    port = 4242
    try:
        port = int(sys.argv[1])
    except Exception as e:
        pass
    return '{}'.format(port)


def main():
    addr = 'tcp://0.0.0.0:' + parse_port()
    s = zerorpc.Server(BackendApi(), heartbeat=100000000)
    s.bind(addr)
    print('start running on {}'.format(addr))

    s.run()


if __name__ == '__main__':
    main()
