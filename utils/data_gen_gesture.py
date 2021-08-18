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
##  generate gesture clips for the MS-G3D from the CSV files    ##
##                                                              ##
##################################################################


MUSICIAN = ["AG", "CC", "SCh"]
LABEL = ["raise", "Other"]


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


def read_label(interval, label_list):
    """
    Parameters
    ----------
    interval int: [start, end]

    label_list : [[name1, start1, end1],
                  [name2, start2, end2],
                  ...]
    Returns
    -------
    label str
    """
    for label in label_list:
        label_interval = [label[1], label[2]]
        iou = compute_iou(interval, label_interval)
        if iou > 0.4:
            if "falls" in label[0]:
                label_name = "Other"
            elif "both" in label[0]:
                label_name = "raise"
            elif "RH" in label[0]:
                label_name = "raise"
            elif "LH" in label[0]:
                label_name = "raise"
            else:
                label_name = "Other"
            break
        else:
            label_name = "Other"
    return label_name


def compute_iou(list1, list2):
    # compute the iou between two intervals
    s1 = list1[0]
    e1 = list1[1]
    s2 = list2[0]
    e2 = list2[1]
    iou_s = max(s1, s2)
    iou_e = min(e1, e2)

    iou = max(iou_e - iou_s, 0) / (e1 - s1)
    return iou


def create_video(output_path, df):
    # create the blank video to check the skeleton
    paths = [
        ["RShoulder", "RElbow"],
        ["LShoulder", "LElbow"],
        ["RElbow", "RWrist"],
        ["LElbow", "LWrist"],
    ]
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    width = height = 500
    out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))
    for ii in range(len(df)):
        img = np.zeros((height, width, 3), np.uint8)
        draw_points(img, df * 500, ii)
        draw_lines(img, df * 500, ii, paths)
        out.write(img)
    out.release()


def draw_points(img, df, frame_index):
    row = df.iloc[frame_index]
    for part in row.index.levels[0]:
        if not np.isnan(row[part]).any():
            pos = (int(row[part]["x"]), int(row[part]["y"]))

            color = (0, 0, 255)
            if part.startswith("R"):
                color = (255, 0, 0)
            elif part.startswith("L"):
                color = (0, 255, 0)

            img = cv2.circle(img, pos, 3, color, -1)


def draw_lines(img, df, frame_index, paths):
    row = df.iloc[frame_index]
    for line in paths:
        pts = np.zeros(shape=(len(line), 2), dtype=np.int32)
        count = 0

        for part in line:
            if part in row.index.levels[0]:
                if (
                    not np.isnan(row[part]).any()
                    and len(np.where(row[part] >= 1)[0]) == 3
                ):
                    pt = np.int32(row[part].values)
                    pts[count] = pt[:2]
                    count += 1

        # Filter out zero points, then reshape before drawing
        pts = pts[pts > 0]
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(
            img,
            [pts],
            False,
            (255, 255, 255),
            thickness=2,
        )


def generate_json(df, label, file):
    """

    Parameters
    ----------
    df : dataframe
        The skeleton data.
    label: str
        The label read from the annotation.
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
    json_data["label"] = label
    json_data["label_index"] = LABEL.index(label)

    json_data["musician"] = file[0]
    json_data["musician_index"] = MUSICIAN.index(file[0])

    for ii in range(len(df)):
        row = df.iloc[ii]
        data = dict()
        data["frame_index"] = ii
        data["skeleton"] = []
        skeleton = dict()
        skeleton["pose"] = [
            row["RElbow"]["x"],
            row["RElbow"]["y"],
            row["RWrist"]["x"],
            row["RWrist"]["y"],
            row["LElbow"]["x"],
            row["LElbow"]["y"],
            row["LWrist"]["x"],
            row["LWrist"]["y"],
        ]

        skeleton["score"] = [
            row["RElbow"]["c"],
            row["RWrist"]["c"],
            row["LElbow"]["c"],
            row["LWrist"]["c"],
        ]
        data["skeleton"].append(skeleton)
        json_data["data"].append(data)
    return json_data


csv_dir = "../data/CSV_Solo/"
annotation_dir = "../data/Annotation/gesture/"
video_dir = "../data/Video_Clips_Solo/"
output_label_dir = "../music_solo_gesture/"
output_train_dir = output_label_dir + "music_solo_train"
output_val_dir = output_label_dir + "music_solo_val"
length = 30
stride = 8


if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)

if not os.path.exists(output_val_dir):
    os.makedirs(output_val_dir)

label_train_data = dict()
label_val_data = dict()

musician_train_data = dict()
musician_val_data = dict()


val_list = ["AG_4a_Nand", "CC_3a_MM"]
train_distribute = [0] * len(LABEL)
test_distribute = [0] * len(LABEL)

for anno in os.listdir(annotation_dir):
    if not anno.endswith(".txt"):
        continue

    file = anno.replace(".txt", "")
    file = file.replace("_Ed", "")
    file = file.replace("_Gestures", "")
    file = file.replace("_Gesture", "")
    csv = file + ".csv"

    # read csv
    dataframe = read_csv(os.path.join(csv_dir, csv))
    num_frame = len(dataframe)

    # read annotation
    with open(os.path.join(annotation_dir, anno), "r") as f:
        label_data = f.readlines()

    label_list = []
    for line in label_data:
        line = line.split("\t")
        start = int(float(line[2]) * 25)
        end = int(float(line[3]) * 25)
        name = line[-1]
        label_list.append([name, start, end])

    # write json files
    json_data = dict()

    for ii, start_frame in enumerate(
        range(0, int(num_frame - length), stride)
    ):
        start_frame += random.randint(-int(stride / 2), int(stride / 2))
        interval = [start_frame, start_frame + length]
        label = read_label(interval, label_list)
        # drop 90% Other sample to balance dataset
        if (label == "Other") and (random.randint(1, 100) <= 86):
            continue
        df = dataframe.iloc[start_frame : start_frame + length]
        json_data = generate_json(df, label, file)

        rand = random.randint(1, 10)

        # random divided into train (80%) or val (20%) set
        if file not in val_list:
            # if rand < 9:
            json_file = os.path.join(
                output_train_dir, file + "_" + str(ii).zfill(5) + ".json"
            )
            output_path = os.path.join(
                output_train_dir,
                file + "_" + str(ii).zfill(5) + "_" + label + ".mp4",
            )
            create_video(output_path, df)
            # write label json
            key = file + "_" + str(ii).zfill(5)

            label_data = dict()
            label_data["has_skeleton"] = True
            label_data["label"] = label
            label_data["label_index"] = LABEL.index(label)
            label_train_data[key] = label_data

            train_distribute[LABEL.index(label)] += 1

        else:

            json_file = os.path.join(
                output_val_dir, file + "_" + str(ii).zfill(5) + ".json"
            )
            output_path = os.path.join(
                output_val_dir,
                file + "_" + str(ii).zfill(5) + "_" + label + ".mp4",
            )
            # if output_path != '../music_solo_gesture/music_solo_val/AG_4a_Nand_00458_RH.mp4':
            #     continue
            create_video(output_path, df)
            # write label json
            key = file + "_" + str(ii).zfill(5)

            label_data = dict()
            label_data["has_skeleton"] = True
            label_data["label"] = label
            label_data["label_index"] = LABEL.index(label)
            label_val_data[key] = label_data

            test_distribute[LABEL.index(label)] += 1

        if os.path.exists(json_file):
            print("file exists, skip")
            continue
        else:
            with open(json_file, "w") as f:
                json.dump(json_data, f)

print(train_distribute)
print(test_distribute)
with open(output_label_dir + "music_solo_train_label.json", "w") as f:
    json.dump(label_train_data, f, indent=4)
with open(output_label_dir + "music_solo_val_label.json", "w") as f:
    json.dump(label_val_data, f, indent=4)
