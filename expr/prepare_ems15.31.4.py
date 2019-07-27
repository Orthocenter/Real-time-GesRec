import sys
sys.path.append('.')

from annotation_ems.gen_dataset import generate_dataset
from annotation_ems.ems_prepare import split, prepare_json

### begin of config

dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
output_path = './annotation_ems'

expr_name = "15.31.4"
modality = "rgb"  # d, rgb, rgbd

# train: first n
train_partition = {
}

# test: all except first n
test_partition = {
    'subject11_01_uneven_padding97108':0,
    'subject11_02_uneven_padding97108':0,
    'subject11_03_uneven_padding97108':0,
    'subject11_04_uneven_padding97108':0,
    'subject11_05_uneven_padding97108':0,
    'subject11_06_uneven_padding97108':0,
    'subject11_07_uneven_padding97108':0,
    'subject11_08_uneven_padding97108':0,
    'subject11_09_uneven_padding97108':0,
    'subject11_10_uneven_padding97108':0,
}

labels = ['wrist_left', 'wrist_right', 'pronation', 'supination']

### end of config

generate_dataset(expr_name=expr_name, modality=modality, dataset_path=dataset_path, output_path=output_path, train_partition=train_partition, test_partition=test_partition, labels=labels, sort_by_filename=True, val_percentage=0.1)

prepare_json(csv_dir_path=output_path, expr_name=expr_name)
