#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil

def move_file(src_path, dst_path, name, rename):
    print('from : {}'.format(src_path))
    print('to : {}'.format(dst_path))
    
    
    f_src = os.path.join(src_path, name)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    f_dst = os.path.join(dst_path, rename)
    if not os.path.exists(f_dst):
        shutil.move(f_src, f_dst)
        # shutil.copyfile(f_src, f_dst)
    else:
        print('the file exists!')
    

def batch_move_files(src_dir, video='overlay', form='.csv'):
    folder_list = os.listdir(src_dir)
    for folder in folder_list:
        if folder == '.DS_Store':
            continue
        if form == '.csv':
            src_path = os.path.join(src_dir, folder)
            file_name = 'person0.csv'
            dst_path = '../CSV_result'
            rename = folder + '.csv'
            if os.path.exists(os.path.join(src_path, file_name)):
                move_file(src_path, dst_path, file_name, rename)
                
        if form == '.avi':
            src_path = os.path.join(src_dir, folder)
            file_name = 'video_' + video +  '.avi'
            dst_path = '../Video_' + video
            rename = folder + '.avi'
            if os.path.exists(os.path.join(src_path, file_name)):
                move_file(src_path, dst_path, file_name, rename)
        
        if form == '.mp4':
            src_path = os.path.join(src_dir, folder)
            file_name = 'video_' + video +  '.mp4'
            dst_path = '../Video_' + video
            rename = folder + '.mp4'
            if os.path.exists(os.path.join(src_path, file_name)):
                move_file(src_path, dst_path, file_name, rename)
                

if __name__ == "__main__":
    src_dir = '../output/video_3d'
    # src_dir = 'output'
    form = '.csv'
    video_form = ''
    batch_move_files(src_dir, video_form, form)

