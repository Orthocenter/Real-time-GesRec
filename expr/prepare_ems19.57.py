import sys
sys.path.append('.')

from annotation_ems.gen_dataset import generate_dataset
from annotation_ems.ems_prepare import split, prepare_json

### begin of config

dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
output_path = './annotation_ems'

expr_name = "19.57"
modality = "rgb"  # d, rgb, rgbd

# train: first n
train_partition = {
    'subject10_02': 100,
    'subject10_03': 100,
    'subject10_04': 100,
    'subject10_05': 100,
    'subject10_06': 100,
    'subject10_07': 100,
}

# test: all except first n
test_partition = {
    'subject10_08': 0,
    'subject10_09': 0,
    'subject10_10': 0,
}

labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

### end of config

generate_dataset(expr_name=expr_name, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels, val_percentage=0.1)

prepare_json(csv_dir_path=output_path, expr_name=expr_name)
