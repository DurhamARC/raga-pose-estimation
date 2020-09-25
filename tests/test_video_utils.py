import os

import cv2

from entimement_openpose.video_utils import crop_video


def test_crop_video():
    # Ensure file doesn't exist
    output_filename = os.path.join("output", "cropped.mp4")
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create a video from the dataframe on a blank background
    x1 = 200
    y1 = 300
    x2 = 800
    y2 = 600
    output_path = crop_video(
        "example_files/example_1person/short_video.mp4",
        "output",
        x1,
        y1,
        x2,
        y2,
    )
    assert output_path == output_filename

    # Check video file has been created
    assert os.path.isfile(output_filename)

    # Check that video size is as we expect
    cap = cv2.VideoCapture(output_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    assert width == x2 - x1
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert height == y2 - y1
    cap.release()
