# tools for data process

The structure of data folder

|------------
|--CITATION.cff
|--data
   |--CSV
      |--CSV_Solo
         |--AG_1a_Jaun.csv
         |--...
      |--CSV_Solo_3d
         |--AG_1a_Jaun.csv
         |--...
      |--...
   |--JSON
      |--AG_1a_Jaun
         |--json
            |--AG_1a_Jaun_000000000_keypoints.json
            |--...
      |--...
   |--Video_Clips_Solo
      |--AG_1a_Jaun.mp4
      |--AG_1b_Jaun.mp4
      |--...
   |--Video_concert
   |--...



I. Regular process
1. video_resize.py 
Afford the function of resizing video into small resolution to reduce the cost of storage.
Afford the function of cutting long video into short clips using opencv (audio will be lost).

2. move_files.py
Afford the function of moving files into another path.

3. bash_ffmpeg_video_cilp.py
Script of cutting video into short clips using ffmpeg (the audio will be reserved)

example:

|--------
|--data
|  |--Video_Clips_Solo
|     |--AG_1a_Jaun.mp4
|     |--...
|--output
   |--video_clips_12s
      |--AG_1a_Jaun
         |--AG_1a_Jaun_000.mp4
         |--AG_1a_Jaun_001.mp4
         |--...



4. bash_add_audio.py
Script of adding the audio into a video file.

example:

|--------
|--data
|  |--Video_overlay
|     |--AG_1a_Jaun.mp4
|     |--...
|  |--Audio
|     |--AG_1a_Jaun.m4a
|     |--...
|--output
   |--Video_overlay_with_sound
      |--AG_1a_Jaun.mp4
      |--...
         

II. Openpose process
1. create_overlay.py
Create overlay videos for single or group video.

example:

|--------
|--data
|  |--Video_Solo_Videos
|     |--AG_1a_Jaun.mp4
|     |--...
|  |--CSV_Solo
|     |--AG_1a_Jaun.csv
|     |--...
|--output
   |--Video_overlay
      |--AG_1a_Jaun.mp4
      |--...

2. create_blank_3d.py
Create the blank video for 3D pose data.

example:

|--------
|--data
|  |--CSV_Solo_3d
|     |--AG_1a_Jaun.csv
|     |--...
|--output
   |--Video_blank_3d
      |--AG_1a_Jaun.mp4
         

3. bash_run_pose_estimation_adaptive/3d.py
Script of running 'run_pose_estimation_adaptive/3d.py' in batch.

III. MS-G3D process
1. data_gen_solo.py
Generate 'MS-G3D form json' files from the CSV file. One json file corresponds to one short clip, including the normalised pose coordinates for 300 frames, label of raga and label of singer.

example:

|--------
|--data
|  |--CSV_Solo_3d
|     |--AG_1a_Jaun.csv
|     |--...
|--music_solo
   |--music_solo_train
      |--AG_1a_Jaun_00000.json
      |--AG_1a_Jaun_00001.json
      |--...
   |--music_solo_val
      |--AG_1b_Jaun_00000.json
      |--AG_1b_Jaun_00001.json
      |--...
   |--music_solo_train_label.json
   |--music_solo_val_label.json


2. data gen_solo_3d.py
Process for the 3D pose data. The data structure is similar.


3. data_split.py
Afford the function of moving specific json samples into train or test set.
Afford the function of generating the new label file for the current train and test set. (It can help when you change samples mutually. Move json files from train/test folder into the other one, and then re-create the label files.)

example:

|--------
|--music_solo
   |--music_solo_train
      |--AG_1a_Jaun_00000.json
      |--AG_1a_Jaun_00001.json
      |--...
   |--music_solo_val
      |--AG_1b_Jaun_00000.json
      |--AG_1b_Jaun_00001.json
      |--...
   |--music_solo_train_label.json
   |--music_solo_val_label.json


