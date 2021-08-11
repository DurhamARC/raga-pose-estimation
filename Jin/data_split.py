import os
from shutil import copyfile
import json


##################################################################
##                                                              ##
##     move files to training and validation folders            ##
##     generate the training and validation label files         ##
##                                                              ##
##################################################################


val_list = ['AG_1b_Jaun', 'AG_2a_Marwa', 'SCh_3b_MM', 'CC_8a_Bag',
            'SCh_6a_Kedar', 'SCh_4b_Nand', 'CC_5a_Shree', 'CC_1b_Bilas',
            'AG_7a_Bahar']

# val_list = ['SCh_1b_Bilas', 'SCh_2b_Jaun', 'SCh_3b_MM', 'SCh_8b_Bag',
#             'SCh_4b_Nand', 'SCh_6b_Kedar', 'SCh_5b_Shree',
#             'SCh_7b_Marwa', 'SCh_9b_Bahar']

def move_files(train_dir, val_dir, all_dir, val_list):
    '''

    Parameters
    ----------
    train_dir : str
        Path to (generated) training folder where the clips are copied to
    val_dir : str
        Path to (generated) validation folder where the clips are copied to
    all_dir : str
        Path to source folder where all clips are located
    val_list : list
        The list of videos that are copied to validation folder
        If empty, all clips are copied to the training folder

    Returns
    -------
    None.

    '''
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    if not len(os.listdir(train_dir)) and not len(os.listdir(val_dir)):
        for file in os.listdir(all_dir):
            
            is_val = False
            for val_name in val_list:
                if file.startswith(val_name):
                    src_file = os.path.join(all_dir, file)
                    dst_file = os.path.join(val_dir, file)
                    copyfile(src_file, dst_file)
                    is_val = True
                    break
            if not is_val:
                src_file = os.path.join(all_dir, file)
                dst_file = os.path.join(train_dir, file)
                copyfile(src_file, dst_file)
            
            print('copy {} to {}'.format(src_file, dst_file))


def generate_label(data_dir):
    '''

    Parameters
    ----------
    data_dir : str
        Path for the root of dataset where train and val folders exist

    Returns
    -------
    None.

    '''
    train_dir = os.path.join(data_dir, 'music_solo_train')
    val_dir = os.path.join(data_dir, 'music_solo_val')
    
    label_train_data = dict()
    label_val_data = dict()
    
    val_list = os.listdir(val_dir)
    val_list.sort()
    for file in val_list: 
        key = file.replace('.json', '')
        with open(os.path.join(val_dir, file), 'r') as f:
            json_data = json.load(f)
            
        label_raga = dict()
        label_raga['has_skeleton'] = True
        label_raga['label'] = json_data['label']
        label_raga['label_index'] = json_data['label_index']
        label_raga['musician'] = json_data['musician']
        label_raga['musician_index'] = json_data['musician_index']
        label_val_data[key] = label_raga
        
    
    train_list = os.listdir(train_dir)
    train_list.sort()
    for file in train_list:
        key = file.replace('.json', '')
        with open(os.path.join(train_dir, file), 'r') as f:
            json_data = json.load(f)
            
        label_raga = dict()
        label_raga['has_skeleton'] = True
        label_raga['label'] = json_data['label']
        label_raga['label_index'] = json_data['label_index']
        label_raga['musician'] = json_data['musician']
        label_raga['musician_index'] = json_data['musician_index']
        label_train_data[key] = label_raga
        
    
    with open(data_dir+'music_solo_train_label.json', 'w') as f:
        json.dump(label_train_data, f, indent=4)
    with open(data_dir+'music_solo_val_label.json', 'w') as f:
        json.dump(label_val_data, f, indent=4)


if __name__ == '__main__':
    move = False
    if move:
        data_dir = '../../MS-G3D/data/music_solo_hand_raw/'
        all_dir = os.path.join(data_dir, 'music_solo_all')
        train_dir = os.path.join(data_dir, 'music_solo_train')
        val_dir = os.path.join(data_dir, 'music_solo_val')
        move_files(train_dir, val_dir, all_dir, val_list)
        
    generate_label('../../MS-G3D/data/music_solo_hand_raw/')