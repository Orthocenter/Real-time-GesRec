import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
import glob

from utils import load_value_file


def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Flow':
                return img.convert('L')
            elif modality == 'Depth':
                return img.convert('L')


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    
    video = []

    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video

    elif modality == 'Depth':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            image_path_depth = os.path.join(video_dir_path.replace('rgb', 'depth'), '{:05d}.jpg'.format(i))
            if os.path.exists(image_path) and os.path.exists(image_path_depth):
                video.append(image_loader(image_path, modality))
                video.append(image_loader(image_path_depth, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key)
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: EMS Dataset - " + subset + " is loading...")
    for i in range(len(video_names)):
    # to test first 3000 only, use this one:
    # for i in range(3000):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        assert(os.path.exists(video_path))

        #n_frames_file_path = os.path.join(video_path, 'n_frames')
        #n_frames = int(load_value_file(n_frames_file_path))
        frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        n_frames = len(frames)

        if n_frames <= 0:
            continue

        begin_t = int(frames[0].split('/')[-1].split('.')[0])
        end_t = begin_t + n_frames - 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': video_names[i]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            raise NotImplementedError()

    return dataset, idx_to_class, class_to_idx




class EMS_shift(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 uneven_gestures_paths,  # path to uneven frames
                 length_configuration,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader,
                 random_offset=0):
        self.data, self.class_names, self.class_to_idx = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

        self.fps = 30
        self.delay = 4 / 30
        self.frame_indices = []
        self.annots = []
        self.num_gestures = 0
        self.num_datasets = 0
        self.index_to_dataset = []
        self.index_to_gesture_id = []
        self.starting_frames = []
        self.uneven_gestures_paths = uneven_gestures_paths
        self.random_offset = random_offset

        self.length_configuration = dict(length_configuration)
        self.init_length_conf()
        self.init_data_paths()
        self.load_annotations()
        self.random_offsets = [0] + [-i for i in range(1,self.random_offset+1)] + [i for i in range(1,self.random_offset+1)]
    
    def find_class_id(self, ges):
        for name, i in self.class_to_idx.items():
            if name in ges:
                return i
        return None
    
    def init_data_paths(self):
        self.data_paths = [os.path.join(p, 'rgb/quick_all') for p in self.uneven_gestures_paths]

    def load_annotaion(self, p):
        with open(p, 'r') as f:
            annot_origin = f.readlines()
            annot_origin = [a for a in annot_origin[0::2]]
        
        annot_origin = annot_origin[10:-10]
        
        annot = []
        frame_indices = []
        starting_frame = []
        for a in annot_origin:
            ges = a.split('start')[0]
            ges = '_'.join(ges.lower().strip().split(' '))
            annot.append(self.find_class_id(ges))
            t = a.split('start:')[-1].strip()
            t = float(t)

            start = int((t + self.delay) * self.fps)
            end = start + self.length_configuration[self.find_class_id(ges)]
            starting_frame.append(len(frame_indices))
            for i in range(start, end):
                frame_indices.append(i)

        self.frame_indices.append(frame_indices)
        self.annots.append(annot)
        self.num_gestures += len(annot)
        self.index_to_dataset += [self.num_datasets for i in range(len(annot))]
        self.index_to_gesture_id += [i for i in range(len(annot))]
        self.starting_frames.append(starting_frame)
        self.num_datasets += 1
    
    def load_annotations(self):
        for p in self.uneven_gestures_paths:
            self.load_annotaion(os.path.join(p, 'quick.txt'))
        print('loaded %d gestures in total' % self.num_gestures)

    def init_length_conf(self):
        keys = list(self.length_configuration.keys())
        for k in keys:
            self.length_configuration[self.find_class_id(k)] = self.length_configuration[k]
        print(self.length_configuration)

    def get_frame_indices(self, dataset_id, gesture_id):
        start = self.starting_frames[dataset_id][gesture_id]
        end = start + self.length_configuration[self.annots[dataset_id][gesture_id]]
        return [self.frame_indices[dataset_id][i] for i in range(start, end)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        dataset_id = self.index_to_dataset[index]
        path = self.data_paths[dataset_id]
        gesture_id = self.index_to_gesture_id[index]

        random_offset = self.random_offsets[random.randint(0, len(self.random_offsets) - 1)]

        ## randomly prepend and append gestures
        gesture_id2 = random.randint(0, len(self.annots[dataset_id]) - 1)
        gesture_id3 = random.randint(0, len(self.annots[dataset_id]) - 1)
        frame_indices = self.get_frame_indices(dataset_id, gesture_id2)
        starting_frame = len(frame_indices)
        frame_indices += self.get_frame_indices(dataset_id, gesture_id) + self.get_frame_indices(dataset_id, gesture_id3)

        start = max(0, starting_frame + random_offset)
        start = min(start, len(frame_indices) - 2)
        end = min(len(frame_indices) - 1, start + 10)

        frame_indices = [frame_indices[i] for i in range(start, end)]
        # print('start', start, 'end', end, 'frame_indices', frame_indices, 'offset', random_offset, 'annot', self.annots[dataset_id][gesture_id])

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)

        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)

        target = (self.annots[dataset_id][gesture_id], torch.tensor((random_offset,)).float())
        
        return clip, target

    def __len__(self):
        return self.num_gestures


