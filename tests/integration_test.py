import os

from entimement_openpose.openpose_json_parser import OpenPoseJsonParser
from entimement_openpose.openpose_parts import OpenPosePartGroups
from entimement_openpose.visualizer import Visualizer


def test_create_overlay():
    path_to_json = os.path.realpath("example_files/example_3people/output_json/")

    # Import Json files
    json_files = [pos_json for pos_json in os.listdir(path_to_json)
                  if pos_json.endswith('.json')]

    # Get array for dataframes
    body_keypoints_dfs = []

    # Loop through all json files in output directory
    # Each file is a frame in the video
    for file in json_files:
        parser = OpenPoseJsonParser(os.path.join(path_to_json, file))
        body_keypoints_df = parser.get_multiple_keypoints(
                                [0, 1],
                                OpenPosePartGroups.UPPER_BODY_PARTS
                            )
        body_keypoints_df.reset_index()
        body_keypoints_dfs.append(body_keypoints_df)

    # Ensure file doesn't exist
    output_filename = 'output/test_short_video_overlay.avi'
    if os.path.exists(output_filename):
        os.remove(output_filename)

    visualizer = Visualizer(output_directory='output')
    visualizer.create_videos_from_dataframes(
        'test_short_video',
        body_keypoints_dfs,
        768,
        576,
        create_blank=False,
        create_overlay=True,
        video_to_overlay='example_files/example_3people/short_video.mp4'
    )

    # Check video file has been created and is about the size we expect
    assert(os.path.isfile(output_filename))
    size = os.path.getsize(output_filename)
    # 7342240
    assert(size > 1000000 and size < 10000000)
