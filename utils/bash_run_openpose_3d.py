#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

##################################################################
##                                                              ##
##    create csv (smoothing jsons) for all 3d pose videos       ##
##                                                              ##
##################################################################

# select a dir with all json files
json_dir = '../data/JSON_hand/'
names = os.listdir(json_dir)
for name in names:
    if name == '.DS_Store':
        continue
    
    # path for the json dir
    input_json = json_dir + name + '/json'  
    # output path
    output_json = '../output/video_hand/' + name
    if os.path.exists(output_json):
        print('{} exists, skip'.format(output_json))
    else:
        print('smooth json from {}'.format(input_json))
        print('output csv into {}'.format(output_json))
        # run the bash command
        command  = 'python3 ../run_openpose_hand.py -j ' + input_json + ' -o ' + output_json
        command += ' -u -c 0.7 -n 2 -s 13 2'
        os.system(command)
    



# os.system(command)
# command = 'runfile('
