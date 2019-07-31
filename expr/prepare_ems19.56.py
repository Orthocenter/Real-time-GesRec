import sys
sys.path.append('.')

from annotation_ems.gen_dataset import generate_dataset
from annotation_ems.ems_prepare import split, prepare_json

### begin of config

dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
output_path = './annotation_ems'

expr_name = "19.56"
modality = "rgb"  # d, rgb, rgbd

# train: first n
train_partition = {
    'subject01_setting3_01': 100,
    'subject01_setting3_02': 100,
    'subject01_setting3_03': 100,
    'subject01_setting3_04': 100,
    'subject01_setting3_05': 100,
    'subject01_setting3_06': 100,
    'subject01_setting3_07': 100,
}

# test: all except first n
test_partition = {
    'subject01_setting3_08': 0,
    'subject01_setting3_09': 0,
    'subject01_setting3_10': 0,
}

labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

### end of config

generate_dataset(expr_name=expr_name, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels, val_percentage=0.1)

prepare_json(csv_dir_path=output_path, expr_name=expr_name)
