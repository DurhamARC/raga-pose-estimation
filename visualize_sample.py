import numpy as np
import cv2
import os

from entimement_openpose.openpose_json_parser import OpenPoseJsonParser
from entimement_openpose.visualization import Visualization

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

# Get array for dataframes
body_keypoints_dfs = []

# Loop through all json files in output directory
# Each file is a frame in the video
# For now, just use the first peson in each video
for file in json_files:
    parser = OpenPoseJsonParser(os.path.join(path_to_json, file))
    body_keypoints_df = parser.get_person_keypoints(0)
    body_keypoints_df.reset_index()
    body_keypoints_dfs.append(body_keypoints_df)

# Draw the data from the DataFrame
for df in body_keypoints_dfs[0:10]:
    img = np.ones((height, width, 3), np.uint8)
    Visualization.draw_lines(img, df)
    Visualization.draw_points(img, df)
    cv2.imshow('image', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
