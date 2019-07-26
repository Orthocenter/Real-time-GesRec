import sys
sys.path.append('.')

from annotation_ems.gen_dataset import generate_dataset
from annotation_ems.ems_prepare import split, prepare_json

### begin of config

dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
output_path = './annotation_ems'

expr_name = "15.30.2"
modality = "rgb"  # d, rgb, rgbd

# train: first n
train_partition = {
    'subject03_03_uneven_padding91078':50,
    'subject03_04_uneven_padding91078':50,
    'subject03_05_uneven_padding91078':50,
    'subject03_06_uneven_padding91078':50,
}

# test: all except first n
test_partition = {
    'subject03_03_uneven_padding91078':50,
    'subject03_04_uneven_padding91078':50,
    'subject03_05_uneven_padding91078':50,
    'subject03_06_uneven_padding91078':50,
    'subject03_07_uneven_padding91078':0,
    'subject03_08_uneven_padding91078':0,
}

labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

### end of config

generate_dataset(expr_name=expr_name, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels, sort_by_filename=True, val_percentage=0.1)

prepare_json(csv_dir_path=output_path, expr_name=expr_name)
