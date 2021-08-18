#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np


###############################################################
##                                                           ##
##    resize the scale of the videos using opencv and clip   ##
##    the videos using opencv (audio will be loss)           ##
##                                                           ##
###############################################################


SCALE = 2  # the scale of resized videos (e.g., new_width = width/SCALE)


def video_resize(input_video_path, output_video_path):
    print("resize video {}".format(input_video_path))
    print("output path {}".format(output_video_path))
    if os.path.exists(output_video_path):
        print("{} exists, skip resize".format(output_video_path))
        return
    else:
        cap = cv2.VideoCapture(input_video_path)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        ret, frame = cap.read()

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (int(width / SCALE), int(height / SCALE)),
        )

        while ret:
            img = cv2.resize(frame, (int(width / SCALE), int(height / SCALE)))
            out.write(img)
            ret, frame = cap.read()

        cap.release()
        out.release()


def video_clip(input_video_path, output_video_dir, length=10):
    print("clip video {}".format(input_video_path))
    print("output dir {}".format(output_video_dir))
    if os.path.exists(output_video_dir):
        print("{} exists, skip cilp".format(output_video_dir))
        return
    else:
        os.mkdir(output_video_dir)
        cap = cv2.VideoCapture(input_video_path)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        ret, frame = cap.read()
        num_frame = cap.get(7)

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

        stride = int(np.ceil(length / 2) * fps)
        for ii, start_frame in enumerate(
            range(0, int(num_frame - length * fps), stride)
        ):
            video_name = input_video_path.split("/")[-1]
            video_name = (
                video_name.split(".")[0] + "_" + str(ii).zfill(3) + ".mp4"
            )
            output_video_path = os.path.join(output_video_dir, video_name)
            out = cv2.VideoWriter(
                output_video_path, fourcc, fps, (int(width), int(height))
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for jj in range(start_frame, start_frame + int(length * fps)):
                ret, frame = cap.read()
                out.write(frame)

            out.release()

        cap.release()


def video_process(input_video_path, output_video_dir, length=10):
    resize_video = input_video_path[:-4] + "_resize.mp4"
    video_resize(input_video_path, resize_video)
    video_clip(resize_video, output_video_dir, length=length)


def batch_video_resize(input_dir, output_dir):
    files = os.listdir(input_dir)
    for file in files:
        if file == ".DS_Store":
            continue
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        input_video = os.path.join(input_dir, file)
        output_video = os.path.join(output_dir, file)
        video_resize(input_video, output_video)


if __name__ == "__main__":
    # video_resize('output/CC_4a/video_blank.avi', 'output/CC_4a/resize_video_blank.mp4')
    # video_resize('output/CC_1b1/video_overlay.avi', 'output/CC_1b1/resize_video_overlap.mp4')
    # video_resize('output/CC_4a/video_blank.avi', 'output/CC_4a/resize_video_blank.mp4')
    # video_clip('output/CC_1b/video_overlay.avi', 'output/CC_1b/overlay_clips_20s/', length=20)
    # video_process('output/CC_1b/video_overlay.avi', 'output/CC_1b/overlay_clips_7s/', length=7)
    # batch_video_resize('../Video_overlay', '../Video_overlay_rest_resize')
    video_resize(
        "../output/video_group/VK_Multani_CentralClose_3601_3719/video_overlay.mp4",
        "../output/video_group/VK_Multani_CentralClose_3601_3719/resize.mp4",
    )
