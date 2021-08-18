#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import cv2
import pandas as pd
import numpy as np


##################################################################
##                                                              ##
## generate clips available for the MS-G3D from the CSV files   ##
## the csv contains one person's 2d skeleton                    ##
##                                                              ##
##################################################################

# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8, "REye"},
# {9, "LEye"},
# {10, "LEye"},
RAGA = [
    "Jaun",
    "Marwa",
    "Bag",
    "Nand",
    "MM",
    "Bilas",
    "Bahar",
    "Kedar",
    "Shree",
]
MUSICIAN = ["AG", "CC", "SCh"]


def read_csv(filename):
    """

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    df_new : dataframe
        Normalised coordinates of 3D pose.

    """
    dataframe = pd.read_csv(filename, index_col="Body Part")
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

        keys = key.split(".")
        if len(keys) == 1:
            key_new = (keys[0], "x")
            xmax = max(xmax, np.max(data_num))
            xmin = min(xmin, np.min(data_num))

        elif len(keys) == 2 and keys[1] == "1":
            key_new = (keys[0], "y")
            ymax = max(ymax, np.max(data_num))
            ymin = min(ymin, np.min(data_num))

        if key == "MidHip":
            data_midhip = data_num
        if key == "Neck":
            data_neck = data_num

    xc = (np.mean(data_neck) + np.mean(data_midhip)) / 2
    width = 2 * max(xc - xmin, xmax - xc)
    height = ymax - ymin

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
                    data[jj] = data[jj - 1]

        keys = key.split(".")
        if len(keys) == 1:
            key_new = (keys[0], "x")
            data = np.round(list((np.array(data) - xmin) / (width)), 5)

        elif len(keys) == 2 and keys[1] == "1":
            key_new = (keys[0], "y")
            data = np.round(list((np.array(data) - ymin) / height), 5)
        else:
            key_new = (keys[0], "c")
            data = np.array(data)

        df[key_new] = data
    df_new = pd.DataFrame(df)
    return df_new


def generate_json(df, file):
    """

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

    """
    file = file.split("_")
    json_data = dict()
    json_data["data"] = []
    json_data["label"] = file[2]
    json_data["label_index"] = RAGA.index(file[2])

    json_data["musician"] = file[0]
    json_data["musician_index"] = MUSICIAN.index(file[0])

    for ii in range(len(df)):
        row = df.iloc[ii]
        data = dict()
        data["frame_index"] = ii
        data["skeleton"] = []
        skeleton = dict()
        skeleton["pose"] = [
            row["Nose"]["x"],
            row["Nose"]["y"],
            row["Neck"]["x"],
            row["Neck"]["y"],
            row["RShoulder"]["x"],
            row["RShoulder"]["y"],
            row["RElbow"]["x"],
            row["RElbow"]["y"],
            row["RWrist"]["x"],
            row["RWrist"]["y"],
            row["LShoulder"]["x"],
            row["LShoulder"]["y"],
            row["LElbow"]["x"],
            row["LElbow"]["y"],
            row["LWrist"]["x"],
            row["LWrist"]["y"],
            row["REye"]["x"],
            row["REye"]["y"],
            row["LEye"]["x"],
            row["LEye"]["y"],
            row["MidHip"]["x"],
            row["MidHip"]["y"],
        ]

        skeleton["score"] = [
            row["Nose"]["c"],
            row["Neck"]["c"],
            row["RShoulder"]["c"],
            row["RElbow"]["c"],
            row["RWrist"]["c"],
            row["LShoulder"]["c"],
            row["LElbow"]["c"],
            row["LWrist"]["c"],
            row["REye"]["c"],
            row["LEye"]["c"],
            row["MidHip"]["c"],
        ]
        data["skeleton"].append(skeleton)
        json_data["data"].append(data)
    return json_data


# path for the CSV source
csv_dir = "../data/CSV_solo/"
# path for the output dir
output_label_dir = "../music_solo/"
output_train_dir = output_label_dir + "music_solo_train"
output_val_dir = output_label_dir + "music_solo_val"
length = 300  # number of frames of each clip
stride = 40  # stride for the start time for each clip


if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)

if not os.path.exists(output_val_dir):
    os.makedirs(output_val_dir)

label_train_data = dict()
label_val_data = dict()


val_list = [
    "AG_1b_Jaun",
    "AG_2a_Marwa",
    "SCh_3b_MM",
    "CC_8a_Bag",
    "SCh_6a_Kedar",
    "SCh_4b_Nand",
    "CC_5a_Shree",
    "CC_1b_Bilas",
    "AG_7a_Bahar",
]
val_ratio = 1  # the ratio of a video file that is used as test

## more challenging only use 2 musicians
# hard
# val_list = ['AG_1a_Jaun', 'AG_1b_Jaun', 'AG_2a_Marwa', 'AG_2b_Marwa',
#             'AG_3a_Bag', 'AG_3b_Bag', 'AG_4a_Nand', 'AG_4b_Nand',
#             'AG_5a_MM', 'AG_5b_MM', 'AG_6a_Bilas', 'AG_6b_Bilas',
#             'AG_7a_Bahar', 'AG_7b_Bahar', 'AG_8_Kedar',
#             'AG_9a_Shree', 'AG_9b_Shree'
#             ]


for csv in os.listdir(csv_dir):
    if csv == ".DS_Store" or "Pakad" in csv:
        continue
    file = csv.split(".")[0]

    # read csv
    dataframe = read_csv(os.path.join(csv_dir, csv))
    num_frame = len(dataframe)

    # write json files
    json_data = dict()

    # random choose a continous duration of the video to be the test sample
    num = len(range(0, int(num_frame - length), stride))
    # random choose the start clip of the continous duration
    rand = random.randint(0, int(num * (1 - val_ratio)))
    val_clip_list = list(range(rand, rand + int(num * val_ratio)))

    # generate json file for each clip
    for ii, start_frame in enumerate(
        range(0, int(num_frame - length), stride)
    ):
        if ii > 0:
            start_frame += random.randint(-int(stride / 2), int(stride / 2))
        df = dataframe.iloc[start_frame : start_frame + length]
        json_data = generate_json(df, file)
        # random divided into train (90%) or val (10%) set
        if not (file in val_list and ii in val_clip_list):
            # data json file
            json_file = os.path.join(
                output_train_dir, file + "_" + str(ii).zfill(5) + ".json"
            )

            # write label json
            key = file + "_" + str(ii).zfill(5)
            # raga and musician
            label_raga = dict()
            label_raga["has_skeleton"] = True
            label_raga["label"] = json_data["label"]
            label_raga["label_index"] = json_data["label_index"]
            label_raga["musician"] = json_data["musician"]
            label_raga["musician_index"] = json_data["musician_index"]
            label_train_data[key] = label_raga

        else:
            json_file = os.path.join(
                output_val_dir, file + "_" + str(ii).zfill(5) + ".json"
            )
            key = file + "_" + str(ii).zfill(5)
            # raga and musician
            label_raga = dict()
            label_raga["has_skeleton"] = True
            label_raga["label"] = json_data["label"]
            label_raga["label_index"] = json_data["label_index"]
            label_raga["musician"] = json_data["musician"]
            label_raga["musician_index"] = json_data["musician_index"]
            label_val_data[key] = label_raga

        if os.path.exists(json_file):
            print("file exists, skip")
            continue
        else:
            with open(json_file, "w") as f:
                json.dump(json_data, f)

with open(output_label_dir + "music_solo_train_label.json", "w") as f:
    json.dump(label_train_data, f, indent=4)
with open(output_label_dir + "music_solo_val_label.json", "w") as f:
    json.dump(label_val_data, f, indent=4)
