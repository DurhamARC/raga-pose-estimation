#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

##################################################################
##                                                              ##
##    create csv (smoothing jsons) for all 2d pose videos       ##
##    do not need to input the number of people in the videos   ##
##                                                              ##
##################################################################

def create_overlay(input_video_path, input_json_path, output_path):
    if os.path.exists(output_path):
        print('video exists, skip')
    else:
        command = 'python3 run_openpose_adaptive.py -j' + input_json_path
        command += ' -v ' + input_video_path
        command += ' -o ' + output_path
        command += ' -u -V -c 0.7 -s 13 2'
        os.system(command)
    # print(command)


if __name__ == "__main__":
    video_list = os.listdir('../data/Video_Solo_Clips/')
    output_dir = '../output/Video_Solo_Clips/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for ii, video in enumerate(video_list):
        if video == '.DS_Store':
            continue

        input_video_path = '../data/Video_Solo_Clips' + video
        input_json_path = '../data/JSON/' + video[:-4] + '/json'
        output_path = output_dir + video
        create_overlay(input_video_path, input_json_path, output_path)       
        

# command = 'ffmpeg -ss 00:00:15 -t 00:00:05 -i output/overlay_with_sound/AG_1a_Jaun.mp4 -vcodec copy -acodec copy output/clip.mp4'
# os.system(command)