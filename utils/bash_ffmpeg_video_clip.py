#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


##################################################################
##                                                              ##
##  cut long video into short clips using ffmpeg                ##
##                                                              ##
##################################################################


def get_time(start, time):
    """

    Parameters
    ----------
    start : float
        Start time (second)
    time : float
        Length of the short clips (second)

    Returns
    -------
    str_ss : str
        Start time with the format hh:mm:ss
    str_t : str
        Length of clips with the format hh:mm:ss

    """
    h = int(np.floor(start / 3600))
    m = int((np.floor(start - h * 3600) / 60))
    s = int(np.floor(start - h * 3600 - m * 60))
    str_ss = str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)

    h = int(np.floor(time / 3600))
    m = int((np.floor(time - h * 3600) / 60))
    s = int(np.floor(time - h * 3600 - m * 60))
    str_t = str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)
    return str_ss, str_t


def video_clip(input_video_path, output_video_dir, length=10):
    """
    Parameters
    ----------
    input_video_path : str
        Path to the long video
    output_video_dir : str
        Path to the dir where generated short clip videos are located
    length : float, optional
        The length of short clips (second)  The default is 10.

    Returns
    -------
    None.

    """
    print("clip video {}".format(input_video_path))
    print("output dir {}".format(output_video_dir))
    if os.path.exists(output_video_dir):
        print("{} exists, skip cilp".format(output_video_dir))
        return
    else:
        os.makedirs(output_video_dir)
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(5)
        num_frame = cap.get(7)
        cap.release()

        stride = int(np.ceil(length * 2 / 3) * fps)

        for ii, start_frame in enumerate(
            range(0, int(num_frame - length * fps), stride)
        ):
            video_name = input_video_path.split("/")[-1]
            video_name = (
                video_name.split(".")[0] + "_" + str(ii).zfill(3) + ".mp4"
            )
            output_video_path = os.path.join(output_video_dir, video_name)

            start_time = int(start_frame / fps)
            str_ss, str_t = get_time(start_time, length)
            command = "ffmpeg -ss " + str_ss + " -t " + str_t
            command += " -i " + input_video_path
            command += (
                " -b:v 10000k -vcodec copy -acodec copy " + output_video_path
            )
            os.system(command)


if __name__ == "__main__":
    length_list = [12]  # generate different lengths of short clips
    video_list = os.listdir("../data/video/")
    for ii, video in enumerate(video_list):
        if video == ".DS_Store":
            continue
        # if ii < 10:
        input_video_path = "../data/video/" + video
        for length in length_list:
            output_video_dir = (
                "../output/video_clips_"
                + length
                + "s/"
                + video[:-4]
                + "_"
                + str(length).zfill(2)
                + "s/"
            )
            video_clip(input_video_path, output_video_dir, length=length)


# command = 'ffmpeg -ss 00:00:15 -t 00:00:05 -i output/overlay_with_sound/AG_1a_Jaun.mp4 -vcodec copy -acodec copy output/clip.mp4'
# os.system(command)
