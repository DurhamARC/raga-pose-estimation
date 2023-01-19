#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

##################################################################
##                                                              ##
##            add audios to the videos created by opencv        ##
##                                                              ##
##################################################################



def add_audio(input_video_path, input_audio_path, output_video_path):
    """

    Parameters
    ----------
    input_video_path : str
        Path of input video with no sound
    input_audio_path : str
        Path of audio
    output_video_path : str
        Path of output video with sound

    Returns
    -------
    None.

    """
    if os.path.exists(output_video_path):
        print("video exists, skip")
    else:
        # run the bash command
        command = "ffmpeg -i " + input_video_path
        command += " -i " + input_audio_path
        command += " -b:v 1500k " + output_video_path
        os.system(command)
    # print(command)


if __name__ == "__main__":
    video_list = os.listdir("../data/Video_overlay/")
    output_dir = "../output/Video_overlay_with_sound/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for ii, video in enumerate(video_list):
        if video == ".DS_Store":
            continue

        input_video_path = "../data/Video/" + video
        input_audio_path = "../data/Audio/" + video[:-4] + ".m4a"
        output_video_path = output_dir + video
        add_audio(input_video_path, input_audio_path, output_video_path)


# command = 'ffmpeg -ss 00:00:15 -t 00:00:05 -i output/overlay_with_sound/AG_1a_Jaun.mp4 -vcodec copy -acodec copy output/clip.mp4'
# os.system(command)
