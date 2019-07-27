import os
cwd = os.getcwd()
import sys
sys.path.append(os.path.abspath("."))
from emulate_uneven_padding import split

data_path = '/mnt/data/yxchen/gesture-datasets/ems/data'

folder_name = [
    'subject01_setting3_01_uneven_padding10789',
    'subject01_setting3_04_uneven_padding10789',
    'subject01_setting3_05_uneven_padding10789',
    'subject01_setting3_06_uneven_padding10789',
    'subject01_setting3_07_uneven_padding10789',
    'subject01_setting3_08_uneven_padding10789',
    'subject01_setting3_09_uneven_padding10789',
    'subject01_setting3_10_uneven_padding10789',

    'subject03_03_uneven_padding91078',
    'subject03_04_uneven_padding91078',
    'subject03_05_uneven_padding91078',
    'subject03_06_uneven_padding91078',
    'subject03_07_uneven_padding91078',
    'subject03_08_uneven_padding91078',

    'subject10_02_uneven_padding81065',
    'subject10_03_uneven_padding81065',
    'subject10_04_uneven_padding81065',
    'subject10_05_uneven_padding81065',
    'subject10_06_uneven_padding81065',
    'subject10_07_uneven_padding81065',
    'subject10_08_uneven_padding81065',
    'subject10_09_uneven_padding81065',
    'subject10_10_uneven_padding81065',

    'subject11_01_uneven_padding97108',
    'subject11_02_uneven_padding97108',
    'subject11_03_uneven_padding97108',
    'subject11_04_uneven_padding97108',
    'subject11_05_uneven_padding97108',
    'subject11_06_uneven_padding97108',
    'subject11_07_uneven_padding97108',
    'subject11_08_uneven_padding97108',
    'subject11_09_uneven_padding97108',
    'subject11_10_uneven_padding97108',
]

padding_configs = {

    'subject01': {
        'quick_pronation': 10,
        'quick_supination': 7,
        'quick_wrist_left': 8,
        'quick_wrist_right': 9,
    },
    'subject03': {
        'quick_pronation': 9,
        'quick_supination': 10,
        'quick_wrist_left': 7,
        'quick_wrist_right': 8,
    },
    'subject10': {
        'quick_pronation': 8,
        'quick_supination': 10,
        'quick_wrist_left': 6,
        'quick_wrist_right': 5,
    },
    'subject11': {
        'quick_pronation': 9,
        'quick_supination': 7,
        'quick_wrist_left': 10,
        'quick_wrist_right': 8,
    },

}

def get_padding_config(folder_name):
    subject = folder_name.split('_')[0]

    return padding_configs[subject]

for i in range(0, len(folder_name)):

    new_folder_name = folder_name[i]
    
    old_folder_name = new_folder_name.split('_uneven_')[0]

    old_folder_path = os.path.join(data_path, old_folder_name)
    new_folder_path = os.path.join(data_path, new_folder_name)

    os.system('mkdir -p %s/rgb' % new_folder_path)
    os.system('cp %s/rgb/quick.mov %s/rgb' % (old_folder_path, new_folder_path))
    os.system('cp %s/quick.txt %s' % (old_folder_path, new_folder_path))
    os.system('rm -r %s/rgb/*_all' % new_folder_path)

    # os.system('python emulate_uneven_padding.py %s' % new_folder_path)
    config = get_padding_config(new_folder_name)
    print(config)
    split(new_folder_path, config)
    os.chdir(cwd)

    # os.system('cp expr/prepare_ems15.29.1.py expr/prepare_ems%s.py' % expr_name[i])
    # cmd = 'sed -i "s/subject01_setting3_08_uneven_padding10789_as_even10/%s/" expr/prepare_ems%s.py' % (folder_name[i], expr_name[i])
    # print(cmd)
    # os.system(cmd)
    # cmd = 'sed -i "s/15.29.1/%s/" expr/prepare_ems%s.py' % (expr_name[i], expr_name[i])
    # print(cmd)
    # os.system(cmd)