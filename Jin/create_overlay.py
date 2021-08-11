#!/usr/bin/env python3

import os
import os.path
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
this_dir = os.path.dirname(__file__)
root_path = os.path.join(this_dir, '..')
add_path(os.path.normpath(root_path))


import pandas as pd
import cv2
from entimement_openpose.visualizer_Jin import Visualizer


##################################################################
##                                                              ##
## create overlay videos for all videos in a folder             ##
##                                                              ##
##################################################################


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
    df = dict()
    for key in dataframe.keys():
        keys = key.split('.')
        if len(keys) == 1:
            key_new = (keys[0], 'x')
        elif len(keys) == 2 and keys[1] == '1':
            key_new = (keys[0], 'y')
        else:
            key_new = (keys[0], 'c')
        
        data = list(dataframe[key][1:])
        data = list(map(float, data))
        df[key_new] = data
    df_new = pd.DataFrame(df)
    return df_new
        
def create_single_overlay(input_video, input_csv, output_dir, 
                          intervals=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('create video into {}'.format(output_dir))
    dataframe = read_csv(input_csv)
    person_dfs = [dataframe]
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(3))
    height = int(cap.get(4))
    cap.release()

    visualizer = Visualizer(output_directory=output_dir)


    print("Creating overlay video...")
    visualizer.create_video_from_dataframes(
        "video",
        person_dfs,
        width,
        height,
        create_overlay=True,
        video_to_overlay=input_video,
        intervals=intervals
    )


def create_group_overlay(input_video, input_csv_dir, output_dir, 
                         intervals=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('create video into {}'.format(output_dir))
    csv_list = os.listdir(input_csv_dir)
    csv_list.sort()
    person_dfs = []
    for csv in csv_list:
        if not csv.endswith('.csv'):
            continue
        input_csv = os.path.join(input_csv_dir, csv)
        dataframe = read_csv(input_csv)
        person_dfs.append(dataframe)
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(3))
    height = int(cap.get(4))
    cap.release()

    visualizer = Visualizer(output_directory=output_dir)


    print("Creating overlay video...")
    visualizer.create_video_from_dataframes(
        "video",
        person_dfs,
        width,
        height,
        create_overlay=True,
        video_to_overlay=input_video,
        intervals=intervals
    )

if __name__ == "__main__":
    single = False
    if single:
        path_list = os.listdir('../data/CSV_Solo/')
        path_list.sort()
        for path in path_list:
            if path == '.DS_Store':
                continue
            name = path.split('.')[0]
            input_video = '../data/Video_Clips_Solo/' + name + '.mp4'
            input_csv = '../data/CSV_Solo/' + path
            output_dir = '../output/overlay/' + name
            # print(output_dir)
            if os.path.exists(output_dir):
                print('output_dir not empty')
            else:
                create_single_overlay(input_video, input_csv, output_dir)
    else:
        path_list = os.listdir('../data/CSV_concert/')
        path_list.sort()
        for path in path_list:
            if path == '.DS_Store':
                continue
            input_video = '../data/Video_concert/' + path + '.mp4'
            input_csv = '../data/CSV_concert/' + path
            output_dir = '../output/overlay_group/'
            if os.path.exists(output_dir):
                print('output_dir not empty')
            else:
                create_group_overlay(input_video, input_csv, output_dir)

