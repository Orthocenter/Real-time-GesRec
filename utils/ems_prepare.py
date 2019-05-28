#%%
import os
import glob
from subprocess import call

def extract_frames():
    """Extract frames of .mov files.
    
    Parameters
    ----------
    """

    files = glob.glob(os.path.join(dataset_path, 
                                    "data",
                                    "*", 
                                    "*.mov")) # this line should be updated according to the full path 
    files += glob.glob(os.path.join(dataset_path, 
                                    "data",
                                    "*",  "*",
                                    "*.mov")) # this line should be updated according to the full path 
    for file in files:
        print("Extracting frames for ", file)
        directory = file.split(".")[0] + "_all"
        if not os.path.exists(directory):
            os.makedirs(directory)
            call(["ffmpeg", "-i",  file, os.path.join(directory, "%05d.jpg"), "-hide_banner"]) 


#%%

import json
import pandas as pd

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels


def convert_csv_to_dict(csv_path, subset, labels):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
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
        train_database = convert_csv_to_dict(train_csv_path, 'training', labels)
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




#%%

dataset_path = '/fastdata/yxchen/gesture-datasets/ems'
# extract_frames()

csv_dir_path = './annotation_ems'
r = '10.3'
label_csv_path = os.path.join(csv_dir_path, 'classInd%s.txt' % r)
train_csv_path = os.path.join(csv_dir_path, 'trainlist%s.txt' % r)
# train_csv_path = None
test_csv_path = os.path.join(csv_dir_path, 'testlist%s.txt' % r)
dst_json_path = os.path.join(csv_dir_path, 'ems%s.json' % r)

convert_jester_csv_to_activitynet_json(
    label_csv_path, train_csv_path, test_csv_path, dst_json_path)