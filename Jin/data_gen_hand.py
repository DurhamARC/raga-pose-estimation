#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import pandas as pd
import numpy as np

##################################################################
##                                                              ##
## generate clips available for the MS-G3D from the CSV files   ##
##                                                              ##
##################################################################


RAGA = ['Jaun', 'Marwa', 'Bag', 'Nand', 'MM', 'Bilas',
        'Bahar', 'Kedar', 'Shree']
MUSICIAN = ['AG', 'CC', 'SCh']

# read and normalise CSV file to the dataframe format
def read_csv(filename):
    '''

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    df_new : dataframe
        Normalised coordinates of 3D pose.

    '''
    dataframe = pd.read_csv(filename, index_col='Body Part')
    # find the bbox of the player for crop
    xmax = 0
    ymax = 0
    xmin = 10000
    ymin = 10000
    
    for key in dataframe.keys():
        data = list(dataframe[key][1:])
        data = list(map(float, data))
        
        data_num = np.array(data)
        data_num = data_num[np.where(~np.isnan(data_num))]
        
        keys = key.split('.')
        if len(keys) == 1:
            key_new = (keys[0], 'x')
            
            xmax = max(xmax, np.max(data_num))
            xmin = min(xmin, np.min(data_num))
            
        elif len(keys) == 2 and keys[1] == '1':
            key_new = (keys[0], 'y')
            ymax = max(ymax, np.max(data_num))
            ymin = min(ymin, np.min(data_num))


    width = (xmax - xmin)
    height = (ymax - ymin)    
    sq = max(width, height)
    width = height = sq
    
    df = dict()
    for key in dataframe.keys():
        data = list(dataframe[key][1:])
        data = list(map(float, data))
        nan_idx = np.where(np.isnan(data))[0]
        if len(nan_idx) == len(data):
            data[:] = 0
        elif len(nan_idx) > 0:
            for jj in nan_idx:
                if jj == 0:
                    data[jj] = np.where(~np.isnan(data))[0][0]
                else:
                    data[jj] = data[jj-1]
                            
        keys = key.split('.')
        if len(keys) == 1:
            key_new = (keys[0], 'x')
            data = np.round(list((np.array(data)-xmin)/(width)), 5)
            
        elif len(keys) == 2 and keys[1] == '1':
            key_new = (keys[0], 'y')
            data = np.round(list((np.array(data)-ymin)/height), 5)
        else:
            key_new = (keys[0], 'c')
            data = np.array(data)
        

        df[key_new] = data
    df_new = pd.DataFrame(df)
    return df_new

# generate data with MS-G3D format
def generate_json(df, file):
    '''

    Parameters
    ----------
    df : dataframe
        The skeleton data.
    file: str
        The file name that provides the label information

    Returns
    -------
    json_data : dict
        The data written to the json file

    '''
    file = file.split('_')
    json_data = dict()
    json_data['data'] = []
    json_data['label'] = file[2]
    json_data['label_index'] = RAGA.index(file[2])
    
    json_data['musician'] = file[0]
    json_data['musician_index'] = MUSICIAN.index(file[0])
    
    for ii in range(len(df)):
        row = df.iloc[ii]
        data = dict()
        data['frame_index'] = ii
        data['skeleton'] = []
        skeleton = dict()
        skeleton['pose'] = [row['LWrist']['x'], row['LWrist']['y'],
                            row['LThumb1']['x'], row['LThumb1']['y'],
                            row['LThumb2']['x'], row['LThumb2']['y'],
                            row['LThumb3']['x'], row['LThumb3']['y'],
                            row['LThumb4']['x'], row['LThumb4']['y'],
                            row['LIndex1']['x'], row['LIndex1']['y'],
                            row['LIndex2']['x'], row['LIndex2']['y'],
                            row['LIndex3']['x'], row['LIndex3']['y'],
                            row['LIndex4']['x'], row['LIndex4']['y'],
                            row['LMiddle1']['x'], row['LMiddle1']['y'],
                            row['LMiddle2']['x'], row['LMiddle2']['y'],
                            row['LMiddle3']['x'], row['LMiddle3']['y'],
                            row['LMiddle4']['x'], row['LMiddle4']['y'],
                            row['LRing1']['x'], row['LRing1']['y'],
                            row['LRing2']['x'], row['LRing2']['y'],
                            row['LRing3']['x'], row['LRing3']['y'],
                            row['LRing4']['x'], row['LRing4']['y'],
                            row['LLittle1']['x'], row['LLittle1']['y'],
                            row['LLittle2']['x'], row['LLittle2']['y'],
                            row['LLittle3']['x'], row['LLittle3']['y'],
                            row['LLittle4']['x'], row['LLittle4']['y'],
                            
                            row['RWrist']['x'], row['RWrist']['y'],
                            row['RThumb1']['x'], row['RThumb1']['y'],
                            row['RThumb2']['x'], row['RThumb2']['y'],
                            row['RThumb3']['x'], row['RThumb3']['y'],
                            row['RThumb4']['x'], row['RThumb4']['y'],
                            row['RIndex1']['x'], row['RIndex1']['y'],
                            row['RIndex2']['x'], row['RIndex2']['y'],
                            row['RIndex3']['x'], row['RIndex3']['y'],
                            row['RIndex4']['x'], row['RIndex4']['y'],
                            row['RMiddle1']['x'], row['RMiddle1']['y'],
                            row['RMiddle2']['x'], row['RMiddle2']['y'],
                            row['RMiddle3']['x'], row['RMiddle3']['y'],
                            row['RMiddle4']['x'], row['RMiddle4']['y'],
                            row['RRing1']['x'], row['RRing1']['y'],
                            row['RRing2']['x'], row['RRing2']['y'],
                            row['RRing3']['x'], row['RRing3']['y'],
                            row['RRing4']['x'], row['RRing4']['y'],
                            row['RLittle1']['x'], row['RLittle1']['y'],
                            row['RLittle2']['x'], row['RLittle2']['y'],
                            row['RLittle3']['x'], row['RLittle3']['y'],
                            row['RLittle4']['x'], row['RLittle4']['y']
                            ]
        
        skeleton['score'] = [row['LWrist']['c'],
                            row['LThumb1']['c'], row['LThumb2']['c'],
                            row['LThumb3']['c'], row['LThumb4']['c'],
                            row['LIndex1']['c'], row['LIndex2']['c'],
                            row['LIndex3']['c'], row['LIndex4']['c'],
                            row['LMiddle1']['c'], row['LMiddle2']['c'],
                            row['LMiddle3']['c'], row['LMiddle4']['c'], 
                            row['LRing1']['c'], row['LRing2']['c'],
                            row['LRing3']['c'], row['LRing4']['c'],
                            row['LLittle1']['c'], row['LLittle2']['c'],
                            row['LLittle3']['c'], row['LLittle4']['c'], 
                            
                            row['RWrist']['c'],
                            row['RThumb1']['c'], row['RThumb2']['c'],
                            row['RThumb3']['c'], row['RThumb4']['c'],
                            row['RIndex1']['c'], row['RIndex2']['c'],
                            row['RIndex3']['c'], row['RIndex4']['c'],
                            row['RMiddle1']['c'], row['RMiddle2']['c'],
                            row['RMiddle3']['c'], row['RMiddle4']['c'], 
                            row['RRing1']['c'], row['RRing2']['c'],
                            row['RRing3']['c'], row['RRing4']['c'],
                            row['RLittle1']['c'], row['RLittle2']['c'],
                            row['RLittle3']['c'], row['RLittle4']['c'], 
                             ]
        data['skeleton'].append(skeleton)
        json_data['data'].append(data)
    return json_data
            

# path for the CSV source
csv_dir = '../data/CSV_hand/'
# path for the output dir
output_label_dir = '../music_solo_hand/'
output_train_dir = output_label_dir + 'music_solo_train'
output_val_dir = output_label_dir + 'music_solo_val'
length = 300 # number of frames of each clip
stride = 40  # stride for the start time for each clip

if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)
    
if not os.path.exists(output_val_dir):
    os.makedirs(output_val_dir)

label_train_data = dict()
label_val_data = dict()

# easy split
val_list = ['AG_1b_Jaun', 'AG_2a_Marwa', 'SCh_3b_MM', 'CC_8a_Bag',
            'SCh_6a_Kedar', 'SCh_4b_Nand', 'CC_5a_Shree', 'CC_1b_Bilas',
            'AG_7a_Bahar']
val_ratio = 1  # the ratio of a video file that is used as test 

## more challenging only use 2 musicians
# hard
# val_list = ['AG_1a_Jaun', 'AG_1b_Jaun', 'AG_2a_Marwa', 'AG_2b_Marwa',
#             'AG_3a_Bag', 'AG_3b_Bag', 'AG_4a_Nand', 'AG_4b_Nand',
#             'AG_5a_MM', 'AG_5b_MM', 'AG_6a_Bilas', 'AG_6b_Bilas',
#             'AG_7a_Bahar', 'AG_7b_Bahar', 'AG_8_Kedar', 'AG_9a_Shree',
#             'AG_9b_Shree'
#             ]


for csv in os.listdir(csv_dir):
    if csv == '.DS_Store':
        continue
    file = csv.split('.')[0]
    
    # read csv    
    dataframe = read_csv(os.path.join(csv_dir, csv))
    num_frame = len(dataframe)
    
    # write json files 
    json_data = dict()
    
    # random choose a continous duration of the video to be the test sample
    num = len(range(0, int(num_frame-length), stride))
    # random choose the start clip of the continous duration
    rand = random.randint(0, int(num*(1-val_ratio)))
    val_clip_list = list(range(rand, rand+int(num*val_ratio)))
    
    # generate json file for each clip
    for ii, start_frame in enumerate(range(0, int(num_frame-length), stride)):
        if ii > 0:
            start_frame += random.randint(-int(stride/2), int(stride/2))
        df = dataframe.iloc[start_frame:start_frame+length]
        json_data = generate_json(df, file)
        if not (file in val_list and ii in val_clip_list):            
            json_file = os.path.join(output_train_dir, 
                                      file+'_'+str(ii).zfill(5)+'.json')
            
            # write label json
            key = file+'_'+str(ii).zfill(5)
            # raga and musician
            label_raga = dict()
            label_raga['has_skeleton'] = True
            label_raga['label'] = json_data['label']
            label_raga['label_index'] = json_data['label_index']
            label_raga['musician'] = json_data['musician']
            label_raga['musician_index'] = json_data['musician_index']
            label_train_data[key] = label_raga

            
        else:
            json_file = os.path.join(output_val_dir, 
                                      file+'_'+str(ii).zfill(5)+'.json')
            key = file+'_'+str(ii).zfill(5)
            # raga and musician
            label_raga = dict()
            label_raga['has_skeleton'] = True
            label_raga['label'] = json_data['label']
            label_raga['label_index'] = json_data['label_index']
            label_raga['musician'] = json_data['musician']
            label_raga['musician_index'] = json_data['musician_index']
            label_val_data[key] = label_raga            
        
        if os.path.exists(json_file):
            print('file exists, skip')
            continue
        else:
            with open(json_file, "w") as f:
                json.dump(json_data, f)

# write label files
with open(output_label_dir+'music_solo_train_label.json', 'w') as f:
    json.dump(label_train_data, f, indent=4)
with open(output_label_dir+'music_solo_val_label.json', 'w') as f:
    json.dump(label_val_data, f, indent=4)
