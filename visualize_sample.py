import cv2
import os

from entimement_openpose.openpose_json_parser import OpenPoseJsonParser
from entimement_openpose.openpose_parts import OpenPosePartGroups
from entimement_openpose.visualizer import Visualizer

# Create a video from JSON point data

# video file
cap = cv2.VideoCapture('example_files/short_video.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
cap.release()

# Paths - should be the folder where Open Pose JSON output was stored
path_to_json = os.path.realpath("example_files/output_json/")

# Import Json files
json_files = [pos_json for pos_json in os.listdir(path_to_json)
              if pos_json.endswith('.json')]
json_files.sort()

# Get array for dataframes
body_keypoints_dfs = []

# Loop through all json files in output directory
# Each file is a frame in the video
for file in json_files:
    parser = OpenPoseJsonParser(os.path.join(path_to_json, file))
    body_keypoints_df = parser.get_multiple_keypoints([0, 1], OpenPosePartGroups.UPPER_BODY_PARTS)
    body_keypoints_df.reset_index()
    body_keypoints_dfs.append(body_keypoints_df)


visualizer = Visualizer(output_directory='output')
visualizer.create_videos_from_dataframes(
    'short_video',
    body_keypoints_dfs,
    width,
    height,
    create_blank=True,
    create_overlay=True,
    video_to_overlay='example_files/short_video.mp4'
)
