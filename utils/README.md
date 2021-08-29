# tools for data process

I. Regular process
1. video_resize.py 
Afford the function of resizing video into small resolution to reduce the cost of storage.
Afford the function of cutting long video into short clips using opencv (audio will be lost).

2. move_files.py
Afford the function of moving files into another path.

3. bash_ffmpeg_video_cilp.py
Script of cutting video into short clips using ffmpeg (the audio will be reserved)

4. bash_add_audio.py
Script of adding the audio into a video file.

II. Openpose process
1. create_overlay.py
Create overlay videos for single or group video.

2. create_blank_3d.py
Create the blank video for 3D pose data.

3. bash_run_openpose_adaptive/3d.py
Script of running 'run_openpose_adaptive/3d.py' in batch.

III. MS-G3D process
1. data_gen_solo.py
Generate 'MS-G3D form json' files from the CSV file. One json file corresponds to one short clip, including the normalised pose coordinates for 300 frames, label of raga and label of singer.

2. data gen_solo_3d.py
Process for the 3D pose data.

3. data_split.py
Afford the function of moving specific json samples into train or test set.
Afford the function of generating the new label file for the current train and test set. (It can help when you change samples mutually. Move json files from train/test folder into the other one, and then re-create the label files.)


